from dpo_trainer import DPOTrainer
import torch
import numpy as np
import math
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch.nn.functional as F
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )

class LinearDecayScheduler:
    def __init__(self, start_value, end_value, start_step, end_step):
        self.start_value = start_value
        self.end_value = end_value
        self.start_step = start_step
        self.end_step = end_step
        self.total_steps = end_step - start_step

    def get_value(self, step):
        if step < self.start_step:
            return float(self.start_value)
        if step > self.end_step:
            return float(self.end_value)
        return self.start_value + (self.end_value - self.start_value) * ((step-self.start_step) / self.total_steps)

class WRPOTrainer(DPOTrainer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Pass all other arguments using **kwargs
        training_args = kwargs["args"]
        num_update_steps_per_epoch = len(self.get_train_dataloader()) // training_args.gradient_accumulation_steps
        self.num_training_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
        if training_args.is_dynamic:
            self.alpha_scheduler = LinearDecayScheduler(start_value=0, end_value=training_args.alpha,
                                                    start_step=0, end_step=self.num_training_steps, )
        else:
            self.alpha = training_args.alpha
        self.loss_type = training_args.loss_type
        self.is_dynamic = training_args.is_dynamic


    @staticmethod
    def tokenize_row(feature, tokenizer,max_length,max_prompt_length,truncation_mode,label_pad_token_id):
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]


        if not isinstance(prompt, str):
            raise ValueError(f"prompt should be an str but got {type(prompt)}")
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

        def build_tokenized_answer(prompt, answer, tokenizer):
            """
            Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
            It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
            Reference:
                https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            """

            full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)
            prompt_input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

            answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids):]
            answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids):]

            # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
            full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

            # Prepare input tokens for token by token comparison
            full_input_ids = np.array(full_tokenized["input_ids"])

            if len(full_input_ids) != len(full_concat_input_ids):
                raise ValueError("Prompt input ids and answer input ids should have the same length.")

            # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
            # can be merged together when tokenizing prompt+answer. This could result
            # on the last token from the prompt being different when tokenized on its own
            # vs when done as prompt+answer.
            response_token_ids_start_idx = len(prompt_input_ids)

            # If tokenized prompt is different than both prompt+answer, then it means the
            # last token has changed due to merging.
            if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
                response_token_ids_start_idx -= 1

            prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
            prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

            if len(prompt_input_ids) != len(prompt_attention_mask):
                raise ValueError("Prompt input ids and attention mask should have the same length.")

            answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
            answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

            return dict(
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                input_ids=answer_input_ids,
                attention_mask=answer_attention_mask,
            )

        if not isinstance(chosen, list):
            raise ValueError(f"chosen should be a list but got {type(chosen)}")
        chosen_tokens_list = [build_tokenized_answer(prompt, chosen_text, tokenizer) for chosen_text in chosen] # list

        if not isinstance(rejected, list):
            raise ValueError(f"rejected should be a list but got {type(rejected)}")
        rejected_tokens_list = [build_tokenized_answer(prompt, rejected_text, tokenizer) for rejected_text in rejected] # list

        # Last prompt token might get merged by tokenizer and
        # it should not be included for generation if that happens
        prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

        chosen_prompt_len_input_ids_list = [len(chosen_tokens["prompt_input_ids"]) for chosen_tokens in chosen_tokens_list]
        rejected_prompt_len_input_ids_list = [len(rejected_tokens["prompt_input_ids"]) for rejected_tokens in rejected_tokens_list]
        prompt_len_input_ids = min(min(chosen_prompt_len_input_ids_list), min(rejected_prompt_len_input_ids_list))

        for k, v in prompt_tokens.items():
            prompt_tokens[k] = v[:prompt_len_input_ids]

        # Make sure prompts only have one different token at most an
        # and length only differs by 1 at most
        for chosen_tokens, chosen_prompt_len_input_ids in zip(chosen_tokens_list,chosen_prompt_len_input_ids_list):
            for rejected_tokens,rejected_prompt_len_input_ids in zip(rejected_tokens_list,rejected_prompt_len_input_ids_list):
                num_diff_tokens = sum(
                    [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
                )
                num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
                if num_diff_tokens > 1 or num_diff_len > 1:
                    raise ValueError(
                        "Chosen and rejected prompt_input_ids might only differ on the "
                        "last token due to tokenizer merge ops."
                    )

        # add EOS token to end of answer
        for chosen_tokens in chosen_tokens_list:
            chosen_tokens["input_ids"].append(tokenizer.eos_token_id)
            chosen_tokens["attention_mask"].append(1)

        for rejected_tokens in rejected_tokens_list:
            rejected_tokens["input_ids"].append(tokenizer.eos_token_id)
            rejected_tokens["attention_mask"].append(1)

        longer_response_length = max(max([len(chosen_tokens["input_ids"]) for chosen_tokens in chosen_tokens_list]), max([len(rejected_tokens["input_ids"]) for rejected_tokens in rejected_tokens_list]))

        # if combined sequence is too long, truncate the prompt
        for answer_tokens in chosen_tokens_list + [prompt_tokens] + rejected_tokens_list :
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > max_length:
                if truncation_mode == "keep_start":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: max_prompt_length]
                elif truncation_mode == "keep_end":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][-max_prompt_length :]
                else:
                    raise ValueError(f"Unknown truncation mode: {truncation_mode}")

        # if that's still too long, truncate the response

        for answer_tokens in chosen_tokens_list + rejected_tokens_list:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > max_length:
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: max_length - max_prompt_length]

        chosen_sequence_tokens_list = []
        for chosen_tokens in chosen_tokens_list:
            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
                label_pad_token_id
            ] * len(chosen_tokens["prompt_input_ids"])
            chosen_sequence_tokens_list.append(chosen_sequence_tokens)

        rejected_sequence_tokens_list = []
        for rejected_tokens in rejected_tokens_list:
            rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
                label_pad_token_id
            ] * len(rejected_tokens["prompt_input_ids"])
            rejected_sequence_tokens_list.append(rejected_sequence_tokens)

        for k, toks in {
            "chosen": chosen_sequence_tokens_list,
            "rejected": rejected_sequence_tokens_list,
            "": prompt_tokens,
        }.items():
            if k != "":
                for type_key in ["input_ids", "attention_mask","labels"]:
                    if type_key == "token_type_ids":
                        continue
                    for ww,tok in enumerate(toks):
                        batch[f"{k}{ww}_{type_key}"]=tok[type_key]
            else:
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens


        return batch


    @staticmethod
    def concatenated_inputs(
            batch: Dict[str, Union[List, torch.LongTensor]],
            label_pad_token_id: int = -100,
            padding_value: int = 0,
            device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        max_length = max(max(batch[f"chosen0_input_ids"].shape[1], batch[f"chosen1_input_ids"].shape[1]),batch[f"rejected0_input_ids"].shape[1])

        for chosen_num in range(2):
            for k in batch:
                if k.startswith(f"chosen{chosen_num}") and isinstance(batch[k], torch.Tensor):
                    if "labels" in k:
                        pad_value = label_pad_token_id
                    elif k.endswith("_input_ids"):
                        pad_value = padding_value
                    elif k.endswith("_attention_mask"):
                        pad_value = 0

                    concatenated_key = k.replace(f"chosen{chosen_num}", "concatenated")
                    if chosen_num == 0:
                        concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
                    else:
                    # switch to list
                        concatenated_batch[concatenated_key] = torch.cat(
                            (
                                concatenated_batch[concatenated_key],
                                pad_to_length(batch[k], max_length, pad_value=pad_value),
                            ),
                            dim=0,
                        ).to(device=device)


        for k in batch:
            if k.startswith(f"rejected0") and isinstance(batch[k], torch.Tensor):
                if "labels" in k:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0

                concatenated_key = k.replace(f"rejected0", "concatenated")
                # switch to list
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        return concatenated_batch

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

        

    def wrpo_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the WRPO loss for a batch of policy model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,num_of_rejected,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the WRPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        chosen_rewards = policy_chosen_logps - reference_chosen_logps
        rejected_rewards = policy_rejected_logps - reference_rejected_logps

        chosen_rewards = chosen_rewards.to(self.accelerator.device)
        rejected_rewards = rejected_rewards.to(self.accelerator.device)

        if self.loss_type == "sigmoid":
            logits = chosen_rewards - rejected_rewards
            losses = (
                    -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
            losses = losses.mean(dim=1)
            weighted_chosen_rewards = chosen_rewards.mean(dim=1)

        elif self.loss_type == "wrpo":
            # alpha under cur training step
            alpha = self.alpha if not self.is_dynamic else self.alpha_scheduler.get_value(self.state.global_step)
            # weight for source and target LLM chosen rewards
            weight = torch.tensor([alpha,1 - alpha], dtype=policy_chosen_logps.dtype, device=policy_chosen_logps.device).view(2,1)
            # count the weighted chosen reward
            weighted_chosen_rewards = torch.matmul(chosen_rewards,weight)
            logits = weighted_chosen_rewards - rejected_rewards
            losses = (
                    -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}."
            )

        chosen_rewards = self.beta * chosen_rewards.detach()
        rejected_rewards = self.beta * rejected_rewards.detach()
        weighted_chosen_rewards = self.beta * weighted_chosen_rewards.detach()

        return losses, chosen_rewards, weighted_chosen_rewards, rejected_rewards

    def concatenated_forward(
            self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )

        bsz = batch["chosen0_labels"].shape[0]  # batch_size

        model_kwargs = {}
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
            label_pad_token_id=self.label_pad_token_id,
        )

        len_chosen = 2 * bsz

        chosen_logps = all_logps[:len_chosen]
        chosen_logps = chosen_logps.view(-1, bsz).t()
        rejected_logps = all_logps[len_chosen:]
        rejected_logps = rejected_logps.view(-1,bsz).t()

        chosen_logits = all_logits[:len_chosen]
        chosen_logits = chosen_logits.view(-1, bsz).t()
        rejected_logits = all_logits[len_chosen:]
        rejected_logits = rejected_logits.view(-1,bsz).t()
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_loss_metrics(
            self,
            model,
            batch: Dict[str, Union[List, torch.LongTensor]],
            train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the WRPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards,weighted_chosen_rewards, rejected_rewards = self.wrpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        target_chosen_logps = policy_chosen_logps[:,-1]
        source_chosen_logps = policy_chosen_logps[:, :-1].mean(dim=1,keepdim=True)
        target_chosen_rewards = chosen_rewards[:,-1]
        source_chosen_rewards = chosen_rewards[:, :-1].mean(dim=1,keepdim=True) 

        source_reward_accuracies = (source_chosen_rewards > rejected_rewards).float()
        pivot_reward_accuracies = (target_chosen_rewards > rejected_rewards).float()
        reward_accuracies = (weighted_chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        if self.is_dynamic:
            cur_alpha = self.alpha_scheduler.get_value(self.state.global_step)
            metrics[f"{prefix}cur_alpha"] = cur_alpha
        metrics[f"{prefix}rewards/chosen"] = weighted_chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/source_chosen"] = source_chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/target_chosen"] = target_chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/source_accuracies"] = source_reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/pivot_accuracies"] = pivot_reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (weighted_chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/source_chosen"] = source_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/target_chosen"] = target_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        return losses.mean(), metrics