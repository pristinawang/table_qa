import re
from grpo import Preprocessor, format_reward_func, accuracy_reward_func
from datasets import load_dataset
import torch
from torch.utils.data import Sampler
from typing import Any, Callable, Optional, Sized, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# def format_reward_func(completions, **kwargs):
#     """Reward function that checks if the completion has a specific format."""
#     # for completion in completions:
#     #     print(type(completion))
#     #     print(completion)
#     #     print(completion[0])
#     #     break
#     pattern = r"^<think>.*?</think><answer>.*?</answer>$"
#     completion_contents = [completion for completion in completions]
#     #matches=[]
#     rewards=[]
#     for content in completion_contents:
#         if re.match(pattern, content):
#             rewards.append(1.0)
#         else:
#             rewards.append(0.0)
#             if "<think>" in content or "</think>" in content or "<answer>" in content or "</answer>" in content:
#                 rewards[-1]=rewards[-1]+0.2

#     #print(matches)
#     #matches = [re.match(pattern, content) for content in completion_contents]
#     return rewards #[1.0 if match else 0.0 for match in matches] #0.5 -0.5

# def accuracy_reward_func(completions, ground_truth, **kwargs):
    

#     matches = [re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL) for completion in completions]
#     contents = [match.group(1) if match else "" for match in matches]
#     # Reward 1 if the content is the same as the ground truth, 0 otherwise
#     return [1.0 if c.lower() == gt.lower() else 0.0 for c, gt in zip(contents, ground_truth)] #-1.0


def test_acc(): #<think>.*?</think><answer>.*?</answer>
    prompts = ["Problem: Solve the equation $2x + 3 = 7$. Solution:", "Problem: Solve the equation $3x - 5 = 10$."]
    completions = [r" The solution is <answer>2</answer> so now it is right", r" The solution is <answer>6</answer> so now"]
    ground_truth = ["2", "5"]
    r=accuracy_reward_func(prompts=prompts, completions=completions, ground_truth=ground_truth)
    print(r)
def test_acc_chat():
    prompts = [
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "assistant", "content": "What is the result of (1 + 2) * 4?"}],
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "assistant", "content": "What is the result of (3 + 1) * 2?"}],
    ]
    
    completions = [
                [{"role": "assistant", "content": " The solution is <answer>2</answer> so now it is right"}],
                [{"role": "assistant", "content": " The solution is <answer>6</answer> so now"}],
            ]
    ground_truth = ["2", "5"]
    r=accuracy_reward_func(prompts=prompts, completions=completions, ground_truth=ground_truth)
    print(r)
def test_format():
    prompts = ["What is the result of (1 + 2) * 4?","What is the result of (3 + 1) * 2?"]
    completions = [
        "here<think>The sum of 1 and 2 is 3, which we multiply by 4 to get 12.</think>space<answer>(1 + 2) * 4 = 12</answer>",
        "<think>The sum of 1 and 2 is 3, which we multiply by 4 to get 12.</think><answer>(1 + 2) * 4 = 12</answer>",
        "The sum of 3 and 1 is 4, which we multiply by 2 to get 8. So (3 + 1) * 2 = 8.",
    ]
    r=format_reward_func(prompts=prompts, completions=completions)
    print(r)
def test_format_chat():
    prompts = [
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "assistant", "content": "What is the result of (1 + 2) * 4?"}],
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "assistant", "content": "What is the result of (3 + 1) * 2?"}],
    ]
    
    completions = [
                [{"role": "assistant", "content": "<think>The sum of 1 and 2 is 3, which we multiply by 4 to get 12.</think><answer>(1 + 2) * 4 = 12</answer>"}],
                [{"role": "assistant", "content": "The sum of 3 and 1 is 4, which we multiply by 2 to get 8. So (3 + 1) * 2 = 8."}],
            ]
    r=format_reward_func(prompts=prompts, completions=completions)
    print(r)
    
def r():
    content="here<think>The sum of 1 and 2 is 3, which we multiply by 4 to get 12.</think>space<answer>(1 + 2) * 4 = 12</answer>"
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"

    matches = re.match(pattern, content)
    print(matches)
    
def test_dataloader():
    dataset = load_dataset("Stanford/wikitablequestions")
    train_dataset = dataset['train']
    train_dataset_small=train_dataset.select(range(10))
    preprocessor = Preprocessor(dataset=train_dataset_small)               
    train_dataset = preprocessor.preprocess()
    train_dataset=train_dataset.remove_columns(['id', 'question', 'answers', 'table'])
    # print('---------------------------------')
    # print(train_dataset[0])
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=3, sampler=RepeatRandomSampler(train_dataset, 3, seed=42))
    print(dataloader)
    
    iterator = iter(dataloader)
    print(iterator)
    first=next(iterator)
    print('---------------------------------')
    print(first)
    second=next(iter(dataloader))
    # print('---------------------------------')
    # print(second)
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
    
def _get_train_sampler() -> Sampler:
    # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
    # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
    # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
    # preventing discrepancies in group formation.
    return RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)

def chat_template():


    # Load the LLaMA instruct model and tokenizer
    model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Change to your model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    # Format input using chat template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]

    # Tokenize using chat format
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=100)

    # Decode response
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response2=tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print('---------1st------------')
    print(response)
    print('------------2nd-------------')
    print(response2)

def nochat_temp():


    # Load the model and tokenizer
    model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Non-chat model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    # Define raw input prompt
    prompt = "What is the capital of France?"

    # Tokenize input (no chat formatting)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=100)

    # Decode response
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response2 = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print('-------1st-----------')
    print(response)
    print('--------2nd---------')
    print(response2)

    
if __name__=="__main__":
    
    test_acc_chat()