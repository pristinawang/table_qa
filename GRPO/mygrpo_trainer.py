import torch
import torch.utils.data
from typing import Any, Callable, Optional, Sized, Union
import numpy as np
import random
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from transformers import (
    GenerationConfig
)
import torch.nn.functional as F
from torch import nn
from accelerate.utils import gather

class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset N times.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`):
            Number of times to repeat each index.
        seed (`Optional[int]`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d"], repeat_count=2)
    >>> list(sampler)
    [2, 2, 0, 0, 3, 3, 1, 1]
    ```
    
    Example usage in GRPO:
    RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.seed)
    """

    def __init__(self, data_source: Sized, repeat_count: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = [
            idx
            for idx in torch.randperm(self.num_samples, generator=self.generator).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count
def get_GRPO_train_dataloader(train_dataset, num_generations, per_device_train_batch_size, seed):
    assert per_device_train_batch_size % num_generations == 0, (
        f"Batch size per device ({per_device_train_batch_size}) must be divisible by "
        f"the number of generations ({num_generations})."
    )

    assert per_device_train_batch_size // num_generations > 0, (
        f"Each batch on a device must at least hold num_generations amount of examples. per_device_train_batch_size can't be zero."
        f"We process {per_device_train_batch_size // num_generations} unique examples per batch on 1 device."
    )
    train_sampler = RepeatRandomSampler(train_dataset, num_generations, seed=seed)
    return DataLoader(train_dataset, batch_size=per_device_train_batch_size, sampler=train_sampler)

class MyGRPOTrainer:
    def __init__(self,ref_model, model, tokenizer, accelerator, reward_funcs, num_generations, per_device_train_batch_size, seed, beta=0.04,max_completion_length=256,generation_temperature=0.9,data_is_conversational=True):
        self.accelerator=accelerator
        self.model=model
        self.max_prompt_length=None
        self.generation_config = GenerationConfig(
                max_new_tokens=max_completion_length,
                do_sample=True,
                temperature=generation_temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
        self.beta=beta
        self.ref_model=ref_model
        self.num_generations=num_generations
        self.per_device_train_batch_size=per_device_train_batch_size
        self.tokenizer=tokenizer
        self.data_is_conversational=data_is_conversational
        self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)
        self.reward_funcs=reward_funcs
        self.seed=seed
        self.set_seed()
        self.init_metrics()
    
    def init_metrics(self):        
        self.custom_metrics = {key: [] for key in [
            'rewards/reward_mean',
            'rewards/reward_std',
            'train-loss',
            'global_grad_norm',
            'advantages/advantage_sum',
            'advantages/advantage_min',
            'advantages/advantage_max',
            'kl',
            'learning_rate',
            'epoch',
            'step',
            'completion_length/average_completion_length',
            'completion_length/max_completion_length'
        ]}

        self.custom_table = {key: [] for key in [
            "epoch", 
            "step", 
            "prompts", 
            "completions", 
            "total_rewards"
        ]}

        for i, reward_func in enumerate(self.reward_funcs):
            self.custom_metrics[f"rewards/{reward_func.__name__}"]=[]
            self.custom_table[f"{reward_func.__name__}"]=[]
    
    def set_seed(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
    
    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        return self._selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens
    
    def _selective_log_softmax(self, logits, index):
        """
        A memory-efficient implementation of the common `log_softmax -> gather` operation.

        This function is equivalent to the following naive implementation:
        ```python
        logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        ```

        Args:
            logits (`torch.Tensor`):
                Logits tensor of shape `(..., num_classes)`.
            index (`torch.Tensor`):
                Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

        Returns:
            `torch.Tensor`:
                Gathered log probabilities with the same shape as `index`.
        """
        if logits.dtype in [torch.float32, torch.float64]:
            selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
            # loop to reduce peak mem consumption
            logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
            per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
        else:
            # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
            per_token_logps = []
            for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
                row_logps = F.log_softmax(row_logits, dim=-1)
                row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
                per_token_logps.append(row_per_token_logps)
            per_token_logps = torch.stack(per_token_logps)
        return per_token_logps
    
    def _inputs_to_device(self, inputs, device):
        for key in inputs.keys(): inputs[key]=inputs[key].to(device)
        return inputs
    
    def prepare_inputs(self,inputs):
        device = self.accelerator.device
        prompts_text = inputs["prompt"]
        prompt_inputs = self.tokenizer(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = self._inputs_to_device(prompt_inputs, device=device)
        print('prompt inputs dict', prompt_inputs.keys())
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        print("prompt_inputs['input_ids'] device:", prompt_inputs["input_ids"].device)
        print("prompt_inputs['attention_mask'] device:", prompt_inputs["attention_mask"].device)

        # inputs = self.tokenizer(prompt, return_tensors="pt").input_ids


        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]



        # No vllm generation, safe to unwrap first before generate. Accelerator may need unwrapping or may not (https://huggingface.co/docs/accelerate/v0.22.0/en/concept_guides/big_model_inference)
        # but deepseed or other accelerator setting may need model to be unwrapped first
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        prompt_completion_ids = unwrapped_model.generate(
                prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
            )

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]
        print("unwrapped_model device:", unwrapped_model.device)
        print("prompt_completion_ids device:", prompt_completion_ids.device)
        # print("prompt length device:", prompt_length.device)
        print("prompt_ids device:", prompt_ids.device)
        print("completion_ids device:", completion_ids.device)
        

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        
        print("is_eos device:", is_eos.device)
        print("eos_idx device:", eos_idx.device)

        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        print("prompt_mask device:", prompt_mask.device)
        print("completion_mask device:", completion_mask.device)
        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                raise ValueError("self.ref_model is None but self.ref_model cannot be None")
        # Decode the generated completions
        completions_text = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        if self.data_is_conversational:
            completions = []
            for completion in completions_text:
                completions.append([{"role": "assistant", "content": completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts_text), len(self.reward_funcs), device=device)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                raise ValueError("reward_func as nn.Module is not supported")
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs.keys() if key not in ["prompt", "completion"]]
                reward_kwargs = {key: inputs[key] for key in keys}
                output_reward_func = reward_func(completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Get the local rewards_per_func for metrics
        local_rewards_per_func = rewards_per_func
        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Caculate local rewards for metrics
        local_rewards = (local_rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        # print('-------mean and std------')
        # print(mean_grouped_rewards)
        # print(std_grouped_rewards)
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        # print('------norm mean and std and adv-------')
        # print(mean_grouped_rewards)
        # print(std_grouped_rewards)
        # print(advantages)
        
        # Get global adv metrics before slicing
        global_adv_sum = advantages.sum().item()
        global_adv_max = advantages.max().item()
        global_adv_min = advantages.min().item()
        
        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts_text),
            (self.accelerator.process_index + 1) * len(prompts_text),
        )
        advantages = advantages[process_slice]
        # print('--------adv-------')
        # print(advantages)
        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        # print('-------new rewards per func--------')
        # print(reward_per_func)
        
        # Only record metrics once
        if self.accelerator.is_main_process:
            # Global metrics: for all processes, global metrics have the same value but we only record global metrics when main_process is being executed
            for i, reward_func in enumerate(self.reward_funcs):
                if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                    raise ValueError("reward_func as nn.Module is not supported")
                else:
                    reward_func_name = reward_func.__name__
                self.custom_metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item()) #global metric for this global step, accelerator.log later
            self.custom_metrics["rewards/reward_mean"].append(rewards.mean().item()) #global
            self.custom_metrics["rewards/reward_std"].append(std_grouped_rewards.mean().item()) #global
            self.custom_metrics["advantages/advantage_sum"].append(global_adv_sum) #global
            self.custom_metrics["advantages/advantage_max"].append(global_adv_max) #global
            self.custom_metrics["advantages/advantage_min"].append(global_adv_min) #global

            # Local metrics: for processes, local metrics are different and we only record the main process's local metrics
            ## Only record main process generations. Avoid using accelerator.gather where it isn't used in trl. Don't want to slow down the code for metrics(not sure if it will but to be safe).
            self.custom_table["prompts"]+= prompts_text
            self.custom_table["completions"]+=completions_text
            self.custom_table["total_rewards"]+=list(local_rewards.detach().cpu().numpy())
            for i, reward_func in enumerate(self.reward_funcs):
                self.custom_table[f"{reward_func.__name__}"]+=list(local_rewards_per_func[:, i].detach().cpu().numpy())


        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages, 
        }
    
    def compute_loss(self, model, inputs, num_items_in_batch=None):

        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # print('-------per token loss1----------')
        # print(per_token_loss)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # print('-------per token loss1----------')
        # print(per_token_loss)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean() # local loss

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        max_completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().max().item()
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # Record metrics only once
        if self.accelerator.is_main_process:
            # global metrics
            self.custom_metrics["completion_length/max_completion_length"].append(max_completion_length) # global
            self.custom_metrics["completion_length/average_completion_length"].append(completion_length) # global
            self.custom_metrics['train-loss'].append(self.accelerator.gather_for_metrics(loss).mean().item()) # global
            self.custom_metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item()) # global
        return loss
    
    def log(self, model, epoch, step, current_lr, max_completion_thresh):

        global_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2) # global 
        if self.accelerator.is_main_process:
            # global metrics
            self.custom_metrics['global_grad_norm'].append(global_grad_norm.item()) # global
            self.custom_metrics['epoch'].append(epoch+1) # global
            self.custom_metrics['step'].append(step+1) #global
            self.custom_metrics['learning_rate'].append(current_lr) # global
        
            # local metrics
            self.custom_table["step"]+=list(np.full((self.per_device_train_batch_size,), step+1))
            self.custom_table["epoch"]+=list(np.full((self.per_device_train_batch_size,), epoch+1))

            # Run to check number of steps  match the number of items in the log
            for log_name,log in self.custom_metrics.items():
                assert (step+1) == len(log), "The number of steps does not match the number of items in the log"

            # Only saving the current step's metrics to dict, step_metrics
            step_metrics = {key: values[-1] for key, values in self.custom_metrics.items()} 
            if step_metrics["completion_length/max_completion_length"]>max_completion_thresh:
                for i in range(-self.per_device_train_batch_size, 0):  # iterate over last batch 
                    self.custom_table["completions"][i] = trainer.custom_table["completions"][i][:max_completion_thresh]
            return step_metrics