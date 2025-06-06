from datasets import load_dataset
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from helper import *
from trl import GRPOConfig, GRPOTrainer
import re
from loadmodel import LoadModel
from peft import LoraConfig, TaskType
import wandb
from custom_grpo import CustomGRPOTrainer
from accelerate import Accelerator, init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from typing import Callable, Optional, Union
from tqdm import tqdm
import numpy as np
from custom_utils import is_conversational
import math
from evaluate import load
from torch.utils.data import DataLoader
from accelerate.utils import is_peft_model
from accelerate.utils import set_seed
import random
from huggingface_hub import snapshot_download
from mygrpo_trainer import MyGRPOTrainer, RepeatRandomSampler, get_GRPO_train_dataloader
import subprocess


def gpu_utilization(gpu_index: int = 0):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
        capture_output=True, text=True
    )
    lines = result.stdout.strip().split('\n')
    used, total = map(int, lines[gpu_index].split(','))
    message=f"GPU {gpu_index} usage: {used / 1024:.2f} GB / {total / 1024:.2f} GB"
    return message


class Preprocessor:
    def __init__(self, dataset, chat, apply_chat_template, tokenizer) -> None:
        '''
        chat is bool: chat format or not
        '''
        self.dataset = dataset
        self.chat=chat
        self.apply_chat_template=apply_chat_template
        self.tokenizer=tokenizer
        #self.system_prompt='A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.'
        self.system_prompt="Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\n\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"
        self.user_prompt_instruction="Return your final response within \\boxed{}."
        
        # "<|begin_of_thought|>\n\nOkay, let me try to figure out this problem. So, we have this operation defined as a⊗b = a²/b. And we need to compute [(1⊗2)⊗3] - [1⊗(2⊗3)]. Then choose the correct answer from the options given. Alright, let's break it down step by step.\n\nFirst, I need to remember that the operation ⊗ is not associative, right? Because the problem is asking for the difference between two different groupings: (1⊗2)⊗3 and 1⊗(2⊗3). So, the order in which we perform the operations matters here. That's probably why there's a subtraction between them.\n\nLet me start by computing each part separately. Let's tackle the first part: (1⊗2)⊗3.\n\nStarting with the innermost operation, which is 1⊗2. According to the definition, a⊗b = a²/b. So here, a is 1 and b is 2. Plugging those in: 1² / 2 = 1/2. So, 1⊗2 equals 1/2.\n\nNow, we take that result and perform the next operation with 3. So, (1⊗2)⊗3 becomes (1/2)⊗3. Again, using the same definition: a is now 1/2 and b is 3. So, ( (1/2)² ) / 3 = (1/4) / 3 = 1/12. So, (1⊗2)⊗3 equals 1/12.\n\nAlright, that's the first part. Now let's compute the second part: 1⊗(2⊗3). Again, starting with the innermost operation, which is 2⊗3. Applying the definition: a is 2 and b is 3. So, 2² / 3 = 4/3. Therefore, 2⊗3 equals 4/3.\n\nNow, we need to compute 1⊗(4/3). Here, a is 1 and b is 4/3. Using the operation definition: 1² / (4/3) = 1 / (4/3) = 3/4. So, 1⊗(2⊗3) equals 3/4.\n\nNow, the problem asks for the difference between the two results: [(1⊗2)⊗3] - [1⊗(2⊗3)] = (1/12) - (3/4). To subtract these fractions, they need a common denominator. The denominators are 12 and 4, so 12 is the common denominator.\n\nConverting 3/4 to twelfths: 3/4 = 9/12. So, 1/12 - 9/12 = (1 - 9)/12 = -8/12. Simplifying that fraction by dividing numerator and denominator by 4: -8/12 = -2/3.\n\nHmm, looking at the answer choices, option A is -2/3. So, is that the answer? Wait, but let me double-check my calculations to make sure I didn't make a mistake somewhere.\n\nFirst, checking (1⊗2): 1² / 2 = 1/2. Correct. Then, (1/2)⊗3: (1/2)² / 3 = (1/4)/3 = 1/12. That seems right.\n\nNow, for 2⊗3: 2² / 3 = 4/3. Correct. Then, 1⊗(4/3): 1² / (4/3) = 1 / (4/3) = 3/4. Yes, that's correct.\n\nSubtracting 3/4 from 1/12: 1/12 - 3/4. Convert 3/4 to 9/12, so 1/12 - 9/12 = -8/12 = -2/3. Yes, that all checks out. So the answer should be -2/3, which is option A.\n\nWait, but let me think again. The operation is defined for all nonzero numbers, so we don't have any issues with division by zero here. 2⊗3 is 4/3, which is fine, and then 1⊗(4/3) is 3/4. Correct.\n\nAlternatively, maybe there's a different way to approach the problem? Let me try expanding both expressions using variables to see if there's a pattern.\n\nLet’s denote the first expression: (a⊗b)⊗c. Using the definition:\n\nFirst, compute a⊗b = a²/b.\n\nThen, take that result and ⊗ with c: (a²/b)⊗c = ( (a²/b)² ) / c = a⁴ / (b² c).\n\nNow, the second expression: a⊗(b⊗c). First compute b⊗c = b²/c.\n\nThen, a⊗(b²/c) = a² / (b²/c) = a² * (c / b²) = (a² c) / b².\n\nTherefore, the difference between the two expressions is:\n\n(a⁴ / (b² c)) - (a² c / b²) = (a⁴ - a² c²) / (b² c) = a² (a² - c²) / (b² c).\n\nHmm, factoring that, it's a² (a - c)(a + c) / (b² c).\n\nBut in our specific problem, a = 1, b = 2, c = 3. Plugging those values in:\n\n1² (1 - 3)(1 + 3) / (2² * 3) = 1 * (-2)(4) / (4 * 3) = (-8) / 12 = -2/3. Same result. So that confirms the answer is indeed -2/3.\n\nTherefore, I think my initial calculation was correct, and the answer is option A.\n\n**Final Answer**\n\\boxed{A}\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\nTo determine the value of \\([(1 \\otimes 2) \\otimes 3] - [1 \\otimes (2 \\otimes 3)]\\) where the operation \\(\\otimes\\) is defined by \\(a \\otimes b = \\frac{a^2}{b}\\), we proceed as follows:\n\nFirst, compute \\(1 \\otimes 2\\):\n\\[\n1 \\otimes 2 = \\frac{1^2}{2} = \\frac{1}{2}\n\\]\nNext, use this result to compute \\((1 \\otimes 2) \\otimes 3\\):\n\\[\n\\left(\\frac{1}{2}\\right) \\otimes 3 = \\frac{\\left(\\frac{1}{2}\\right)^2}{3} = \\frac{\\frac{1}{4}}{3} = \\frac{1}{12}\n\\]\n\nNow, compute \\(2 \\otimes 3\\):\n\\[\n2 \\otimes 3 = \\frac{2^2}{3} = \\frac{4}{3}\n\\]\nThen, use this result to compute \\(1 \\otimes (2 \\otimes 3)\\):\n\\[\n1 \\otimes \\left(\\frac{4}{3}\\right) = \\frac{1^2}{\\frac{4}{3}} = \\frac{1}{\\frac{4}{3}} = \\frac{3}{4}\n\\]\n\nFinally, find the difference between the two results:\n\\[\n\\frac{1}{12} - \\frac{3}{4} = \\frac{1}{12} - \\frac{9}{12} = \\frac{1 - 9}{12} = \\frac{-8}{12} = -\\frac{2}{3}\n\\]\n\nThus, the answer is \\(\\boxed{A}\\).\n\n<|end_of_solution|>"

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
            formatted_user_input = self.user_prompt_instruction+"\n"+'Table:\n'+TableToPIPE(T=example['table'])+'\nQuestion: '+example['question']
            formatted_sys_input = self.system_prompt
            formatted_messages = [
                        {"role": "system", "content": formatted_sys_input},
                        {"role": "user", "content": formatted_user_input}
                    ]
            formatted_output = process_answers(answers=example["answers"])
            return {"prompt": formatted_messages, "ground_truth": formatted_output}
        elif self.chat and self.apply_chat_template:
            formatted_user_input = self.user_prompt_instruction+"\n"+'Table:\n'+TableToPIPE(T=example['table'])+'\nQuestion: '+example['question']
            formatted_sys_input = self.system_prompt
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
    
    # pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    ## pattern that has \n\n in between each line
    # pattern = (
    #     r"^<\|begin_of_thought\|\>\n\n.*?<\|end_of_thought\|\>\n\n<\|begin_of_solution\|\>\n\n.*?<\|end_of_solution\|\>$"
    # )
    pattern = (
        r"^<\|begin_of_thought\|\>.*?<\|end_of_thought\|\>\n\n<\|begin_of_solution\|\>.*?<\|end_of_solution\|\>$"
    )
    
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
    # pattern = r"<answer>(.*?)</answer>"
    pattern = r"<\|begin_of_solution\|>.*?\\boxed\{(.*?)\}.*?<\|end_of_solution\|>"
    
    if isinstance(completions[0],str):
        completion_contents = completions  
    else:
        completion_contents = [completion[0]["content"] for completion in completions]
    
    matches = [re.search(pattern, completion, re.DOTALL) for completion in completion_contents]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c.lower() == gt.lower() else 0.0 for c, gt in zip(contents, ground_truth)] #-1.0

def print_trainable_parameters(model, accelerator=None):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if accelerator is not None:
        accelerator.print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
    else:
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

def print_trainable_parameters_names(model):
    trainable_params = [(name, param.shape) for name, param in model.named_parameters() if param.requires_grad]

    print("Trainable parameters:")
    for name, shape in trainable_params:
        print(f" - {name}: {shape}")

## Pytorch train loop
# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html


def main():
    # Accelerator
    accelerator = Accelerator(log_with="wandb")
    # accelerator.init_trackers("tableQA-GRPO")
    
    # # Job id
    # now = datetime.now()
    # job_id = now.strftime("%Y%m%d%H%M%S")
    # accelerator.print('----job_id------')
    # accelerator.print(job_id)
    # accelerator.print('------------------')

    
    
    # ## Set seed
    # seed=42
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed)
    # set_seed(seed)
    
    # ## Check if we have cuda?
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # accelerator.print("Using", device, "; using gpu:", torch.cuda.get_device_name())
    
    # ## Dataset
    # dataset = load_dataset("Stanford/wikitablequestions")
    # train_dataset = dataset['train']
    # val_dataset = dataset['validation']
    # test_dataset  = dataset['test']
    # train_dataset_small=train_dataset.select(range(4))
    # val_dataset_small=val_dataset.select(range(120))
    
    accelerator.print(f"Before loading model1 : {gpu_utilization(0)}")
    accelerator.print(f"Before loading model1 : {gpu_utilization(1)}")
    
    ## Load models
    model_id="bespokelabs/Bespoke-Stratos-7B"#'meta-llama/Meta-Llama-3-8B-Instruct'#"meta-llama/Llama-3.1-8B-Instruct" #'meta-llama/Meta-Llama-3-8B-Instruct' #"meta-llama/Llama-3.2-1B-Instruct"
    # weights_location = snapshot_download(repo_id=model_id)
    # model_config = AutoConfig.from_pretrained(model_id)
    # with init_empty_weights():
    #     empty_model = AutoModelForCausalLM.from_config(model_config)
    # accelerator.print('--------saved model names start with---------')
    # accelerator.print("tableQA-GRPO-"+model_id.split('/')[1]+"-"+job_id)
    # accelerator.print('----------------------------------')
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # peft_config = LoraConfig(task_type='CAUSAL_LM', inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    # bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, llm_int8_threshold = 6)
    # accelerator.print(f"After empty model : {gpu_utilization(0)}")
    # accelerator.print(f"After empty model : {gpu_utilization(1)}")
    model = AutoModelForCausalLM.from_pretrained(model_id) 
    accelerator.print(f"Before prepare model : {gpu_utilization(0)}")
    accelerator.print(f"Before prepare model : {gpu_utilization(1)}")
    model = accelerator.prepare(model)
    # {"": 0}
    #model = load_and_quantize_model(empty_model, device_map='auto', weights_location=weights_location, bnb_quantization_config=bnb_quantization_config)
    #model.add_adapter(peft_config, adapter_name="adapter_lora")
    # print_trainable_parameters(model, accelerator=accelerator)
    ## Special tokens
    # tokenizer.pad_token = tokenizer.eos_token
    accelerator.print(f"After preparing model1 : {gpu_utilization(0)}")
    accelerator.print(f"After preparing model1 : {gpu_utilization(1)}")
    ## Reference Model
    # ref_model = AutoModelForCausalLM.from_pretrained(model_id)
    # with init_empty_weights():
    #     empty_ref_model = AutoModelForCausalLM.from_config(model_config)
    # ref_model = load_and_quantize_model(empty_ref_model, device_map='auto', weights_location=weights_location, bnb_quantization_config=bnb_quantization_config)
    # accelerator.print(f"After loading model2: {gpu_utilization(0)}")
    # accelerator.print(f"After loading model2: {gpu_utilization(1)}")
    
    ## GRPO Config
    # mygrpo_config = {
    #     "num_generations":3,
    #     "per_device_train_batch_size":3,
    #     "num_gpus":2,
    #     "max_completion_thresh":700,
    #     "num_epoch":1
    #         }
    
    # ## Data preparation
    # preprocessor = Preprocessor(dataset=train_dataset_small, chat=True, apply_chat_template=True, tokenizer=tokenizer)               
    # train_dataset = preprocessor.preprocess()
    # train_dataset=train_dataset.remove_columns(['id', 'question', 'answers', 'table'])


    # global_train_batch_size = mygrpo_config['per_device_train_batch_size'] * mygrpo_config['num_gpus'] # We want len(train_dataset) % global_train_batch_size == 0, if not, accelerator will use repeated data items to path the final batch
    # # train_sampler = RepeatRandomSampler(train_dataset, mygrpo_config['num_generations'], seed=seed)
    # # train_dataloader = DataLoader(train_dataset, batch_size=mygrpo_config['per_device_train_batch_size'], sampler=train_sampler)
    # train_dataloader=get_GRPO_train_dataloader(train_dataset=train_dataset, num_generations=mygrpo_config['num_generations'], per_device_train_batch_size=mygrpo_config['per_device_train_batch_size'], seed=seed)
    
    ## Optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    # total_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    # accelerator.print(f"Total parameters in optimizer: {total_params}")
    # ## Accelerator
    # train_dataloader, optimizer = accelerator.prepare(train_dataloader, optimizer)

    # grpo=MyGRPOTrainer(
    #             tokenizer=tokenizer, 
    #             accelerator=accelerator, 
    #             reward_funcs=[format_reward_func, accuracy_reward_func],
    #             ref_model=ref_model,
    #             num_generations=mygrpo_config['num_generations'],
    #             model=model,
    #             per_device_train_batch_size=mygrpo_config['per_device_train_batch_size'],
    #             seed=seed
    #         ) # lora, sync ref model
    
    # for epoch in range(mygrpo_config['num_epoch']):
    #     for step,batch in enumerate(train_dataloader):
    #         model.train()
    #         inputs=grpo.prepare_inputs(inputs=batch)
    #         print('-------------DEBUG inputs')
    #         print(inputs)
    #         print('-----------------')
    #         loss=grpo.compute_loss(model=model, inputs=inputs)
    #         print('------loss-------')
    #         print(loss)
    #         print('------step metrics-------')
    #         accelerator.backward(loss)
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         del inputs
    #         torch.cuda.empty_cache()
    #         step_metrics=grpo.log(model=model, epoch=epoch, step=step, current_lr=optimizer.param_groups[0]['lr'], max_completion_thresh=mygrpo_config['max_completion_thresh'])
    #         print(step_metrics)
    #         accelerator.log(step_metrics)
    # accelerator.end_training()


def main_working():
    # Job id
    # now = datetime.now()
    # job_id = now.strftime("%Y%m%d%H%M%S")
    # print('----job_id------')
    # print(job_id)
    # print('------------------')
    
    # # Wandb
    # wandb.login()
    # run = wandb.init(
    #     # Set the project where this run will be logged
    #     project="tableQA-GRPO"
        
    # )
    
    # ## Set seed
    # seed=42
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed)
    
    # ## Check if we have cuda?
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Using", device, "; using gpu:", torch.cuda.get_device_name())
    
    # ## Dataset
    # dataset = load_dataset("Stanford/wikitablequestions")
    # train_dataset = dataset['train']
    # val_dataset = dataset['validation']
    # test_dataset  = dataset['test']
    # train_dataset_small=train_dataset.select(range(11))
    # val_dataset_small=val_dataset.select(range(120))
    
    
    
    ## Load models
    model_id="bespokelabs/Bespoke-Stratos-7B"#'meta-llama/Meta-Llama-3-8B-Instruct'#"meta-llama/Llama-3.1-8B-Instruct" #'meta-llama/Meta-Llama-3-8B-Instruct' #"meta-llama/Llama-3.2-1B-Instruct"
    # print('--------saved model names start with---------')
    # print("tableQA-GRPO-"+model_id.split('/')[1]+"-"+job_id)
    # print('----------------------------------')
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    peft_config = LoraConfig(task_type='CAUSAL_LM', inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    accelerator = Accelerator()
    print(f"After acc init : {gpu_utilization(0)}")
    print(f"After acc init : {gpu_utilization(1)}")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.add_adapter(peft_config, adapter_name="adapter_lora")
    # print_trainable_parameters(model)
    ## Special tokens
    # tokenizer.pad_token = tokenizer.eos_token
    
    ## Reference Model
    # loadmodel = LoadModel(pretrained_model=model_id, tune_type='4bit', device='auto')  #4bit #AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
    # ref_model = loadmodel.load_model()
    
    ## Data preparation
    # preprocessor = Preprocessor(dataset=train_dataset_small, chat=True, apply_chat_template=True, tokenizer=tokenizer)               
    # train_dataset = preprocessor.preprocess()
    # train_dataset=train_dataset.remove_columns(['id', 'question', 'answers', 'table'])

    # train_batch_size=120
    # train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    
    # ## Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    print(f"Before prepare model : {gpu_utilization(0)}")
    print(f"Before prepare model : {gpu_utilization(1)}")
    ## Accelerator
    # accelerator = Accelerator()
    
    model, optimizer = accelerator.prepare(model, optimizer)
    print(f"After prepare : {gpu_utilization(0)}")
    print(f"After prepare : {gpu_utilization(1)}")
        



if __name__=="__main__":
    ## log the group reward so that you can see if format is getting importance or if acc is getting more attention
    ## Prompt needs to be changed as well. Maybe only question and no format since it's supposed to learn on its own
    main_working()
