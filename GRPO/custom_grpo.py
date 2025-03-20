from trl import GRPOConfig, GRPOTrainer
from typing import Callable, Optional, Union
from collections import defaultdict
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    Trainer
)
from torch import nn
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
import torch
from packaging import version
from datasets import Dataset, IterableDataset
from transformers.utils import is_peft_available
if is_peft_available():
    from peft import PeftConfig
from custom_utils import *
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]



    
class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_metrics={}
        self.custom_metrics['rewards/reward_mean']=[]
        self.custom_metrics['rewards/reward_std']=[]
        for i, reward_func in enumerate(self.reward_funcs):
            self.custom_metrics[f"rewards/{reward_func.__name__}"]=[]
        self.custom_metrics["completion_length"]=[]
        self.custom_metrics['train-loss']=[]
        self.custom_metrics['global_grad_norm']=[]
        self.custom_metrics["advantages/advantage_sum"]=[]
        self.custom_metrics["advantages/advantage_min"]=[]
        self.custom_metrics["advantages/advantage_max"]=[]
        self.custom_metrics["kl"]=[]
        self.custom_metrics['learning_rate']=[]
        self.custom_metrics['epoch']=[]
        self.custom_metrics['step']=[]
        self.custom_table={
                            "epoch": [],
                            "step": [],
                            "prompts": [],
                            "completions": [],
                            "total_rewards": [],
                            "format_rewards": [],
                            "accuracy_rewards": []
                        }
            
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        print('--------Prompt, first item----------', len(prompts_text))
        for prompt in prompts_text:
            print(prompt)
            break
        print('---------prp----------')
        # print('------Has or not------')
        # print(hasattr(Trainer, "_prepare_inputs"))  
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        print('---completions_text, first completion----', len(completions_text))
        print(completions_text[0])
        print('--------------')
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
        print('---Comple, first completion----', len(completions))
        for completion in completions:
            print(completion)
            break
        print('--------------')
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        print('------rewards per func--------')
        print(rewards_per_func)
        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        print('----rewards------')
        print(rewards)
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        print('-------mean and std------')
        print(mean_grouped_rewards)
        print(std_grouped_rewards)
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        print('------norm mean and std and adv-------')
        print(mean_grouped_rewards)
        print(std_grouped_rewards)
        print(advantages)
        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        print('--------adv-------')
        print(advantages)
        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        print('-------new rewards per func--------')
        print(reward_per_func)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
            print(reward_func_name, reward_per_func[i].item())
            # if not(f"rewards/{reward_func_name}" in self.custom_metrics): self.custom_metrics[f"rewards/{reward_func_name}"]=[reward_per_func[i].item()]
            # else:self.custom_metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
            self.custom_metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        
        
        self.custom_table["prompts"]+= prompts_text
        self.custom_table["completions"]+=completions_text
        self.custom_table["format_rewards"]+=list(rewards_per_func[:, 0].detach().cpu().numpy())
        self.custom_table["accuracy_rewards"]+=list(rewards_per_func[:, 1].detach().cpu().numpy())
        self.custom_table["total_rewards"]+=list(rewards.detach().cpu().numpy())
        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())
        self.custom_metrics["rewards/reward_mean"].append(rewards.mean().item())
        self.custom_metrics["rewards/reward_std"].append(std_grouped_rewards.mean().item())
        self.custom_metrics["advantages/advantage_sum"].append(advantages.sum().item())
        self.custom_metrics["advantages/advantage_max"].append(advantages.max().item())
        self.custom_metrics["advantages/advantage_min"].append(advantages.min().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # print('------inputs--------')
        # print(inputs)
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        # print('-----per token logps-------')
        # print(per_token_logps)
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
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # print('---intermed---------')
        # print((per_token_loss * completion_mask))
        # print((per_token_loss * completion_mask).sum(dim=1))
        # print(((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)))
        # print(((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean())
        print('-------loss----------')
        print(loss)
        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        self.custom_metrics["completion_length"].append(completion_length)

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self.custom_metrics['train-loss'].append(loss.item())
        self.custom_metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        return loss
    def batch_eval(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        print('--------Prompt, first item----------', len(prompts_text))
        for prompt in prompts_text:
            print(prompt)
            break
        print('---------prp----------')
        # print('------Has or not------')
        # print(hasattr(Trainer, "_prepare_inputs"))  
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        print('---completions_text, first completion----', len(completions_text))
        print(completions_text[0])
        print('--------------')
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
        print('---Comple, first completion----', len(completions))
        for completion in completions:
            print(completion)
            break
        print('--------------')
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        print('------rewards per func--------')
        print(rewards_per_func)
        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        print('----rewards------')
        print(rewards)
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        print('-------mean and std------')
        print(mean_grouped_rewards)
        print(std_grouped_rewards)
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        print('------norm mean and std and adv-------')
        print(mean_grouped_rewards)
        print(std_grouped_rewards)
        print(advantages)
        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        print('--------adv-------')
        print(advantages)
        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        print('-------new rewards per func--------')
        print(reward_per_func)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
            print(reward_func_name, reward_per_func[i].item())
            # if not(f"rewards/{reward_func_name}" in self.custom_metrics): self.custom_metrics[f"rewards/{reward_func_name}"]=[reward_per_func[i].item()]
            # else:self.custom_metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
            self.custom_metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        
        
        self.custom_table["prompts"]+= prompts_text
        self.custom_table["completions"]+=completions_text
        self.custom_table["format_rewards"]+=list(rewards_per_func[:, 0].detach().cpu().numpy())
        self.custom_table["accuracy_rewards"]+=list(rewards_per_func[:, 1].detach().cpu().numpy())
        self.custom_table["total_rewards"]+=list(rewards.detach().cpu().numpy())
        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())
        self.custom_metrics["rewards/reward_mean"].append(rewards.mean().item())
        self.custom_metrics["rewards/reward_std"].append(std_grouped_rewards.mean().item())
        self.custom_metrics["advantages/advantage_sum"].append(advantages.sum().item())
        self.custom_metrics["advantages/advantage_max"].append(advantages.max().item())
        self.custom_metrics["advantages/advantage_min"].append(advantages.min().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
    
    
# class CustomGRPOTrainer(GRPOTrainer):
#     def __init__(
#         self, log_with=None, accelerator=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         if log_with=="wandb": 
#             self.log_with=log_with
#             self.accelerator=accelerator
    
#     def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
#         metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

#         # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
#         # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
#         if next(iter(logs.keys())).startswith("eval_"):
#             metrics = {f"eval_{key}": val for key, val in metrics.items()}

#         logs = {**logs, **metrics}
#         if self.log_with=='wandb':
#             self.accelerator.log(logs, step=self.state.global_step)
        
#         super(GRPOTrainer, self).log(logs, start_time)
#         self._metrics.clear()
        
## Example usage
# peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
# grpo_args = GRPOConfig(output_dir="./out", logging_steps=10) #, use_vllm=True, vllm_device='cuda:0') #, vllm_gpu_memory_utilization=0.1

# trainer = CustomGRPOTrainer(
#     model=model,
#     processing_class=tokenizer,
#     reward_funcs=[format_reward_func, accuracy_reward_func],
#     args=grpo_args,
#     train_dataset=train_dataset,
#     peft_config=peft_config,
#     log_with='wandb',
#     accelerator=accelerator
# )
# trainer.train()