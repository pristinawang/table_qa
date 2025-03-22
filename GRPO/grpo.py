from datasets import load_dataset
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from helper import *
from trl import GRPOConfig, GRPOTrainer
import re
from loadmodel import LoadModel
from peft import LoraConfig, TaskType
import wandb
from custom_grpo import CustomGRPOTrainer
from accelerate import Accelerator
from typing import Callable, Optional, Union
from tqdm import tqdm
import numpy as np
from custom_utils import is_conversational
import math
from evaluate import load
from torch.utils.data import DataLoader
class Preprocessor:
    def __init__(self, dataset, chat, apply_chat_template, tokenizer) -> None:
        '''
        chat is bool: chat format or not
        '''
        self.dataset = dataset
        self.chat=chat
        self.apply_chat_template=apply_chat_template
        self.tokenizer=tokenizer
    #     self.tokenizer=tokenizer
    # def tokenize_function(self, examples):
    #     return self.tokenizer(text=examples["prompt"], text_target=examples['completion'], padding=True) 

    def format_input(self, example):
        """
        Concatenates the question and table columns as a single input sequence.
        
        Args:
            example (dict): A single dataset entry.

        Returns:
            dict: Formatted example with `input_text` and `word_labels`.
        """
        if self.chat and not(self.apply_chat_template):
            # {"prompt": [{"role": "user", "content": "What color is the sky?"}]}
            formatted_user_input = 'Table:\n'+TableToPIPE(T=example['table'])+'\nQuestion: '+example['question']
            formatted_sys_input = 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.'
            formatted_messages = [
                        {"role": "system", "content": formatted_sys_input},
                        {"role": "user", "content": formatted_user_input}
                    ]
            formatted_output = process_answers(answers=example["answers"])
            return {"prompt": formatted_messages, "ground_truth": formatted_output}
        elif self.chat and self.apply_chat_template:
            formatted_user_input = 'Table:\n'+TableToPIPE(T=example['table'])+'\nQuestion: '+example['question']
            formatted_sys_input = 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.'
            formatted_messages = [
                        {"role": "system", "content": formatted_sys_input},
                        {"role": "user", "content": formatted_user_input}
                    ]
            formatted_output = process_answers(answers=example["answers"])
            example = {"prompt": formatted_messages, "ground_truth": formatted_output}
            

            ## From trainer code: apply_chat_template
            if "prompt" in example:
                last_role = example["prompt"][-1]["role"]
                if last_role == "user":
                    add_generation_prompt = True
                    continue_final_message = False
                elif last_role == "assistant":
                    add_generation_prompt = False
                    continue_final_message = True
                else:
                    raise ValueError(f"Invalid role in the last message: {last_role}")
                prompt = self.tokenizer.apply_chat_template(
                    example["prompt"],
                    #tools=tools,
                    continue_final_message=continue_final_message,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            return {"prompt": prompt, "ground_truth": formatted_output}
        else:
            formatted_input = '\nPlease provide your reasoning process within <think>...</think> tags and only include the final answer inside <answer>...</answer> tags.\nTable:\n'+TableToPIPE(T=example['table'])+'\nQuestion: '+example['question']
            formatted_output = process_answers(answers=example["answers"])
            return {"prompt": formatted_input, "ground_truth": formatted_output}

    def preprocess(self):
        """
        Applies formatting and tokenization to the dataset.

        Returns:
            Dataset: The tokenized dataset.
        """
        # Apply formatting function
        formatted_dataset = self.dataset.map(self.format_input) #load_from_cache_file=False
        return formatted_dataset

def finegrained_format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # for completion in completions:
    #     print(type(completion))
    #     print(completion)
    #     print(completion[0])
    #     break
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion for completion in completions]
    #matches=[]
    rewards=[]
    for content in completion_contents:
        if re.match(pattern, content):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            # for spec_tok in ["<think>", "</think>", "<answer>", "</answer>"]:
            #     if spec_tok in content:
            #         rewards[-1]=rewards[-1]+0.1
            if "<think>" in content or "</think>" in content or "<answer>" in content or "</answer>" in content:
                rewards[-1]=rewards[-1]+0.2

    #print(matches)
    #matches = [re.match(pattern, content) for content in completion_contents]
    return rewards #[1.0 if match else 0.0 for match in matches] #0.5 -0.5
def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    
    # print('----Format reward, Comple Format::---')
    # print(completions)
    # print('----------------')
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    if isinstance(completions[0],str):
        completion_contents = completions  
    else:
        completion_contents = [completion[0]["content"] for completion in completions]
    # print('-----Format reward, formatted completion-------')
    # print(completion_contents)
    # print('-------------------')
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches] #0.5 -0.5
    

def accuracy_reward_func(completions, ground_truth, **kwargs):
    # print('----Acc reward, Comple Format::---')
    # print(completions)
    # print('----------------')
    if isinstance(completions[0],str):
        completion_contents = completions  
    else:
        completion_contents = [completion[0]["content"] for completion in completions]
    
    # print('-----Acc reward, formatted completion-------')
    # print(completion_contents)
    # print('-------------------')
    matches = [re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL) for completion in completion_contents]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c.lower() == gt.lower() else 0.0 for c, gt in zip(contents, ground_truth)] #-1.0

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
def main():
    # Job id
    now = datetime.now()
    job_id = now.strftime("%Y%m%d%H%M%S")
    print('----job_id------')
    print(job_id)
    print('------------------')
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="tableQA-GRPO"
        
    )
    
    
    ####
    # accelerator = Accelerator(log_with="wandb")

    # accelerator.init_trackers("tableQA-GRPO")

    ####

    ## Check if we have cuda?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device, "; using gpu:", torch.cuda.get_device_name())
    
    ## Dataset
    dataset = load_dataset("Stanford/wikitablequestions")
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset  = dataset['test']
    train_dataset_small=train_dataset.select(range(11))
    val_dataset_small=val_dataset.select(range(11))
    # Use this to test multiple answers: train_dataset_small=train_dataset.select(range(6250,6251))
    
    ## Load models
    model_id='meta-llama/Meta-Llama-3-8B-Instruct'#"meta-llama/Llama-3.1-8B-Instruct" #'meta-llama/Meta-Llama-3-8B-Instruct' #"meta-llama/Llama-3.2-1B-Instruct"
    print('--------saved model names start with---------')
    print("tableQA-GRPO-"+model_id.split('/')[1]+"-"+job_id)
    print('----------------------------------')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    loadmodel = LoadModel(pretrained_model=model_id, tune_type='4bit', device='auto')  #4bit #AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
    model = loadmodel.load_model()
    
    ## Special tokens
    tokenizer.pad_token = tokenizer.eos_token
    
    # print('OG Dataset')
    # print(train_dataset_small[0])
    
    
    
    preprocessor = Preprocessor(dataset=train_dataset, chat=True, apply_chat_template=False, tokenizer=None)               
    train_dataset = preprocessor.preprocess()
    train_dataset=train_dataset.remove_columns(['id', 'question', 'answers', 'table'])
    val_batch_size=8
    preprocessor = Preprocessor(dataset=val_dataset, chat=True, apply_chat_template=True, tokenizer=tokenizer)               
    val_dataset = preprocessor.preprocess()
    val_dataset=val_dataset.remove_columns(['id', 'question', 'answers', 'table'])
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)
    val_iterator = iter(val_dataloader)
    # print('Final dataset')
    # print(train_dataset[0])
    peft_config = LoraConfig(task_type='CAUSAL_LM', inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    ## this LoraConfig isn't needed here. LoadModel already has LoraConfig
    grpo_args = GRPOConfig(
        output_dir="./out", 
        logging_steps=1,
        num_generations=3,
        per_device_train_batch_size=6,
        seed=42
    ) #, use_vllm=True, vllm_device='cuda:0') #, vllm_gpu_memory_utilization=0.1
    if grpo_args.per_gpu_train_batch_size:
        print('GPU batch:', grpo_args.per_gpu_train_batch_size)
    else:
        print('device:', grpo_args.per_device_train_batch_size)
    print("Args:", grpo_args.train_batch_size)
    trainer = CustomGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward_func, accuracy_reward_func],
        args=grpo_args,
        train_dataset=train_dataset,
        #peft_config=peft_config
    )
    # trainer = GRPOTrainer(
    #     model=model,
    #     processing_class=tokenizer,
    #     reward_funcs=[format_reward_func, accuracy_reward_func],
    #     args=grpo_args,
    #     train_dataset=train_dataset,
    #     peft_config=peft_config
    # )
    # print('----Trainer?----')
    # print(type(trainer))
    num_processes = trainer.accelerator.num_processes
    global_batch_size = grpo_args.per_device_train_batch_size * num_processes
    possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
    # print(num_processes, global_batch_size)
    # print(possible_values)
    # print('------data set, first item----')
    # for i in range(len(train_dataset)):
    #     print(train_dataset[i])
    #     break
    # print('------------------')
    # for step, inputs in tqdm(enumerate(cust_trainer.get_train_dataloader())):
    #     print('Step:', step)
    #     print(inputs)
    #     # del inputs[0]["prompt"]
    #     # print(inputs)
    #     inputs = cust_trainer._prepare_inputs(inputs)
    #     print('------INPUTS-----------')
    #     print(inputs)
    #     print('-------------------')
    #     loss = cust_trainer.compute_loss(model, inputs)
    #     print('-------LOSS-----------')
    #     print(loss)
    #     print('------------------')
    # print("Batch size", trainer._train_batch_size)
    # trainer.train()
    
    device = trainer.args.device
    model = trainer.model.to(device)
    model.train()  # Set to training mode

    # Get optimizer and scheduler (created by Trainer)
    optimizer = trainer.optimizer if trainer.optimizer else torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = trainer.lr_scheduler if trainer.lr_scheduler else None
    
    train_dataloader = trainer.get_train_dataloader()
    
    # Print DataLoader parameters
    #print('------Iter-----------')
    train_iterator = iter(train_dataloader)
    print("Train Batch size", trainer._train_batch_size)
    print('# Steps in Train loop', len(train_dataloader))
    print("Val Batch size", val_batch_size)
    print('# Steps in Val loop', len(val_dataloader))
    print('-------# Trainable Para------')
    print_trainable_parameters(model=model)
    print('----------------------------')
    # for step in enumerate(range(4)):
    #     batch=next(train_iterator)
    #     print('------INPUTS-----------')
    #     print('Step')
    #     print(step)
    #     print('Batch')
    #     print(batch)
    # print('-----TRAIN-----------')

    
    num_epochs=1
    check_point_step=math.ceil(len(train_dataloader)/10)
    eval_freq=50
    print('------Eval Freq: every ? train_step-------')
    print(eval_freq)
    print('-------------------------------------------')
    metric = load("exact_match")
    for epoch in range(num_epochs):
        model.train()
        for step in tqdm(range(len(train_dataloader))):
            inputs=next(train_iterator)
            model.zero_grad()
            #print('Step:',step)
            # print('------Bef Inputs-------')
            # print(inputs)
            inputs = trainer._prepare_inputs(inputs)
            # print('------INPUTS Length-----------')
            # print(inputs['prompt_ids'].shape[0])
            batch_num = inputs['prompt_ids'].shape[0]
            loss = trainer.compute_loss(model, inputs)
            # print('-------LOSS-----------')
            # print(loss)
            del inputs
            torch.cuda.empty_cache()
            trainer.accelerator.backward(loss)
            #print('Loss:', loss)
            loss=loss.detach()
            #print('loss detatch', loss)
            optimizer.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            ## Log
            global_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2)
            # print('----GRAD-----')
            # print(global_grad_norm)
            trainer.custom_metrics['global_grad_norm'].append(global_grad_norm.item())
            trainer.custom_metrics['epoch'].append(epoch+1)
            trainer.custom_metrics['step'].append(step+1)
            trainer.custom_metrics['learning_rate'].append(current_lr)
            trainer.custom_table["step"]+=list(np.full((batch_num,), step+1))
            trainer.custom_table["epoch"]+=list(np.full((batch_num,), epoch+1))
            #print('---log----')
            for log_name,log in trainer.custom_metrics.items():
                assert (step+1) == len(log), "The number of steps does not match the number of items in the log"
                #print(log_name, log)
            
            df = pd.DataFrame(trainer.custom_table)
            # Dictionary to store the last items
            step_metrics = {key: values[-1] for key, values in trainer.custom_metrics.items()}
            step_metrics["completions"]=wandb.Table(dataframe=df)

            run.log(step_metrics)
            # run.log({
            #     "epoch": epoch + 1,
            #     # "step": step + 1,
            #     "train-loss": loss,
            #     "learning_rate": current_lr
            # })
            if (step%eval_freq==0 and step!=0) or step >=len(train_dataloader)-1:
                model.eval()  # Set to training mode
                val_iterator = iter(val_dataloader)
                #print('----Val loop', step/eval_freq,'-----------')
                for val_step in tqdm(range(len(val_dataloader))):
                    inputs=next(val_iterator)
                    # print('-----My Val loader------')
                    # print(inputs)
                    # print('------------------------')
                    
                    completions = trainer.batch_eval(inputs=inputs, is_conversational_qa=True)
                    predictions=completions_to_answers(completions=completions)
                    # print('------completions-----------')
                    # print(completions)
                    # print('------predictions--------')
                    # print(predictions)
                    metric.add_batch(predictions=predictions, references=inputs['ground_truth'])
                results=metric.compute()
                # print('------eval results------')
                # print(results)
                run.log({
                    "eval/accuracy":results["exact_match"],
                    "eval/step": step/eval_freq
                })
    
            if (step%check_point_step==0 and step!=0) or step >=len(train_dataloader)-1:
                model.push_to_hub("tableQA-GRPO-"+model_id.split('/')[1]+"-"+job_id+"-step"+str(step))
    
def test_acc(): #<think>.*?</think><answer>.*?</answer>
    prompts = ["Problem: Solve the equation $2x + 3 = 7$. Solution:", "Problem: Solve the equation $3x - 5 = 10$."]
    completions = [r" The solution is <answer>2</answer> so now it is right", r" The solution is <answer>6</answer> so now"]
    ground_truth = ["2", "5"]
    r=accuracy_reward_func(prompts=prompts, completions=completions, ground_truth=ground_truth)
    print(r)

def test_format():
    prompts = [
        [{"role": "assistant", "content": "What is the result of (1 + 2) * 4?"}],
        [{"role": "assistant", "content": "What is the result of (3 + 1) * 2?"}],
    ]
    completions = [
        [{"role": "assistant", "content": "<think>The sum of 1 and 2 is 3, which we multiply by 4 to get 12.</think><answer>(1 + 2) * 4 = 12</answer>"}],
        [{"role": "assistant", "content": "<think>The sum of 1 and 2 is 3, which we multiply by 4 to get 12.</think><answer>(1 + 2) * 4 = 12</answer>"}],
        [{"role": "assistant", "content": "The sum of 3 and 1 is 4, which we multiply by 2 to get 8. So (3 + 1) * 2 = 8."}],
    ]
    r=format_reward_func(prompts=prompts, completions=completions)
    print(r)

if __name__=="__main__":
    ## log the group reward so that you can see if format is getting importance or if acc is getting more attention
    ## Prompt needs to be changed as well. Maybe only question and no format since it's supposed to learn on its own
    main()
    
