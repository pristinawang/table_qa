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
from torch.utils.data import DataLoader
from grpo import Preprocessor


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


    ## Check if we have cuda?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device, "; using gpu:", torch.cuda.get_device_name())
    print('TEST')
    ## Dataset
    dataset = load_dataset("Stanford/wikitablequestions",trust_remote_code=True)
    print('TEST2')
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset  = dataset['test']
    print('Size of Dataset')
    print('Train:', len(train_dataset))
    print('Val:', len(val_dataset))
    print('Test:', len(test_dataset))
    train_dataset_small=train_dataset.select(range(3))
    # Use this to test multiple answers: train_dataset_small=train_dataset.select(range(6250,6251))
    
    ## Load models
    model_id='meta-llama/Meta-Llama-3-8B-Instruct'#"meta-llama/Llama-3.1-8B-Instruct" #'meta-llama/Meta-Llama-3-8B-Instruct' #"meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    loadmodel = LoadModel(pretrained_model=model_id, tune_type='4bit', device='auto')  #4bit #AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
    model = loadmodel.load_model()
    
    ## Special tokens
    tokenizer.pad_token = tokenizer.eos_token
    
    # print('OG Val Dataset')
    # print(val_dataset[0])
    preprocessor = Preprocessor(dataset=val_dataset, chat=True, apply_chat_template=True)               
    val_dataset = preprocessor.preprocess()
    val_dataset=val_dataset.remove_columns(['id', 'question', 'answers', 'table'])
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)
    val_iterator = iter(val_dataloader)
    print('First Iter')
    batch_val_data=next(val_iterator)
    print(batch_val_data['ground_truth'])
    print(type(batch_val_data))
    print(len(batch_val_data['ground_truth']))
    batch_val_data=next(val_iterator)
    print('2nd Iter')
    print(batch_val_data['ground_truth'])
    print(type(batch_val_data))
    print(len(batch_val_data['ground_truth']))
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
        train_dataset=val_dataset,
        peft_config=peft_config
    )

    # num_processes = trainer.accelerator.num_processes
    # global_batch_size = grpo_args.per_device_train_batch_size * num_processes
    # possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]

    
    device = trainer.args.device
    model = trainer.model.to(device)
    model.eval()  # Set to training mode


    ##
    train_Useval_dataloader = trainer.get_train_dataloader()
    train_iterator = iter(train_Useval_dataloader) 
    inputs=next(train_iterator)
    print('----train loader----')
    print(inputs)
    print('---------------------')


    inputs=next(val_iterator)
    print('-----My Val loader------')
    print(inputs)
    print('------------------------')
    inputs = trainer._prepare_inputs(inputs)
    print('------INPUTS Length-----------')
    print(inputs['prompt_ids'].shape[0])
    batch_num = inputs['prompt_ids'].shape[0]
    loss = trainer.compute_loss(model, inputs)

    # Get optimizer and scheduler (created by Trainer)
    # optimizer = trainer.optimizer if trainer.optimizer else torch.optim.AdamW(model.parameters(), lr=5e-5)
    # scheduler = trainer.lr_scheduler if trainer.lr_scheduler else None
    
    
    
    # num_epochs=1
    # check_point_step=math.ceil(len(train_dataloader)/10)
    # for epoch in range(num_epochs):
    #     for step in tqdm(range(len(train_dataloader))):
    #         inputs=next(train_iterator)
    #         model.zero_grad()
    #         print('Step:',step)
    #         # print('------Bef Inputs-------')
    #         # print(inputs)
    #         inputs = trainer._prepare_inputs(inputs)
    #         print('------INPUTS Length-----------')
    #         print(inputs['prompt_ids'].shape[0])
    #         batch_num = inputs['prompt_ids'].shape[0]
    #         loss = trainer.compute_loss(model, inputs)
    #         print('-------LOSS-----------')
    #         print(loss)
    #         del inputs
    #         torch.cuda.empty_cache()
    #         trainer.accelerator.backward(loss)
    #         print('Loss:', loss)
    #         loss=loss.detach()
    #         print('loss detatch', loss)
    #         optimizer.step()
    #         current_lr = optimizer.param_groups[0]['lr']
            
    #         ## Log
    #         global_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2)
    #         print('----GRAD-----')
    #         print(global_grad_norm)
    #         trainer.custom_metrics['global_grad_norm'].append(global_grad_norm.item())
    #         trainer.custom_metrics['epoch'].append(epoch+1)
    #         trainer.custom_metrics['step'].append(step+1)
    #         trainer.custom_metrics['learning_rate'].append(current_lr)
    #         trainer.custom_table["step"]+=list(np.full((batch_num,), step+1))
    #         trainer.custom_table["epoch"]+=list(np.full((batch_num,), epoch+1))
    #         print('---log----')
    #         for log_name,log in trainer.custom_metrics.items():
    #             assert (step+1) == len(log), "The number of steps does not match the number of items in the log"
    #             print(log_name, log)
            
    #         df = pd.DataFrame(trainer.custom_table)
    #         # Dictionary to store the last items
    #         step_metrics = {key: values[-1] for key, values in trainer.custom_metrics.items()}
    #         step_metrics["completions"]=wandb.Table(dataframe=df)



    #         # if (step%check_point_step==0 and step!=0) or step >=len(train_dataloader)-1:
    #         #     model.push_to_hub("tableQA-GRPO-"+model_id.split('/')[1]+"-"+job_id+"-step"+str(step))
    


if __name__=="__main__":
    ## log the group reward so that you can see if format is getting importance or if acc is getting more attention
    ## Prompt needs to be changed as well. Maybe only question and no format since it's supposed to learn on its own
    main()
    
