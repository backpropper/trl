import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.modeling_utils import unwrap_model
from .sft_trainer import SFTTrainer
from ..import_utils import is_peft_available

if is_peft_available():
    from peft import PeftModel


class IWSFTTrainer(SFTTrainer):
    def __init__(
        self,
        model,
        args,
        train_dataset,
        eval_dataset,
        dataset_text_field,
        max_seq_length,
        peft_config,
        padding_side,
        trl_eval_size,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            max_seq_length=max_seq_length,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field=dataset_text_field,
            peft_config=peft_config,
            padding_side=padding_side,
            trl_eval_size=trl_eval_size,
        )
        self.__dict__.update(kwargs)
        if self.accelerator.num_processes == 1:
            self.trl_device = 0 if torch.cuda.is_available() else "cpu"

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # if self.label_smoother is not None and "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        #     labels = None

        # generate_with_grad = undecorated(unwrap_model(model).generate)
        # unwrap_model(model).generate_with_grad = MethodType(
        # generate_with_grad, unwrap_model(model)
        # )
        outputs = unwrap_model(model).generate(
            **inputs,
            **self.trl_generation_kwargs,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=False,
        )
        
        main_loss = 0
        kl_loss = None
        total_kl = torch.tensor(0.0, device=self.trl_device)
        responses = []
        eos = torch.tensor(0.0, device=self.trl_device)
        for i in range(len(inputs["input_ids"])):
            response = outputs.sequences[i, inputs["input_ids"][i].shape[-1] :]
            last_inds = torch.nonzero(response == self.tokenizer.eos_token_id)
            last_ind = -1
            if len(last_inds) != 0:
                last_ind = int(last_inds[0])
                if last_ind == 0:
                    print("Warning: empty response")
                    print(response)
                    print(outputs.sequences[i])
                    print(response[:last_ind+1])
                    eos += 1
                    loss = torch.tensor(float('nan'))
                    return (loss, total_kl, total_kl, total_kl, total_kl, eos)
                response = response[:last_ind+1]
            responses.append(response)
            target = response
            logits = outputs.scores[: response.shape[-1]]
            logits = torch.stack(logits).permute(1, 0, 2)[i]
            per_loss = F.cross_entropy(logits, target.long())
            main_loss += per_loss
            if self.kl_enable:
                with torch.no_grad():
                    with unwrap_model(self.model).disable_adapter():
                        ref_logits = (
                            self.model(input_ids=response.unsqueeze(0))["logits"].detach().squeeze()
                        )
                per_rev_kl = F.kl_div(
                    torch.log_softmax(ref_logits, -1),
                    torch.log_softmax(logits, -1),
                    log_target=True,
                    reduction="none",
                ).sum(-1)
                per_kl = F.kl_div(
                    torch.log_softmax(ref_logits, -1),
                    torch.log_softmax(logits, -1),
                    log_target=True,
                    reduction="none",
                ).sum(-1)
                if kl_loss is None:
                    kl_loss = (per_kl + per_rev_kl ) / 2
                else:
                    kl_loss = torch.cat((kl_loss, (per_kl + per_rev_kl) / 2), dim=0)
        if self.kl_enable:
            total_kl = torch.mean(kl_loss)

        with torch.no_grad():
            answers = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

            query = self.tokenizer.batch_decode(
                inputs["input_ids"], skip_special_tokens=True
            )
            texts = [q + r for q, r in zip(query, answers)]
            ds = Dataset.from_dict({"text": texts})
            kds = KeyDataset(ds, "text")
            pipe_outputs = self.trl_sentiment_pipe(kds, **self.trl_sent_kwargs)
            rewards = [
                torch.tensor(
                    (output[0]["score"] - self.trl_rew_mean) / self.trl_rew_std, device=self.trl_device
                )
                for output in pipe_outputs
            ]
            rewards = torch.stack(rewards).mean()

            iw = self.trl_alpha * (1 - torch.sigmoid(self.trl_beta * rewards))
            
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # if labels is not None:
        #     if is_peft_available() and isinstance(model, PeftModel):
        #         model_name = unwrap_model(model.base_model)._get_name()
        #     else:
        #         model_name = unwrap_model(model)._get_name()
        #     if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #         loss = self.label_smoother(outputs, labels, shift_labels=True)
        #     else:
        #         loss = self.label_smoother(outputs, labels)
        # else:
        #     if isinstance(outputs, dict) and "loss" not in outputs:
        #         raise ValueError(
        #             "The model did not return a loss from the inputs, only the following keys: "
        #             f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
        #         )
        #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #     loss = (
        #         outputs["loss"] * iw if isinstance(outputs, dict) else outputs[0] * iw
        #
        loss = main_loss * iw + self.kl_weight * total_kl
        return (
            (loss, outputs)
            if return_outputs
            else (loss, main_loss, total_kl, iw, rewards, eos)
        )

    # def save_pretrained(self, save_directory, *args, **kwargs):
    #     self.accelerator.unwrap_model(self.model).save_pretrained(save_directory)
    #     self.tokenizer.save_pretrained(save_directory)
