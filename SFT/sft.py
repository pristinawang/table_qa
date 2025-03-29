import os, sys
import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_scheduler, AdamW
from tqdm import tqdm
# Add the GRPO folder (sibling of SFT) to the path
grpo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'GRPO'))
sys.path.append(grpo_path)

# Now import from loadmodel.py
from loadmodel import *

def data_test():
    dataset = load_dataset('bespokelabs/Bespoke-Stratos-17k')
    print(dataset)
    train_dataset = dataset['train']
    train_dataset_small=train_dataset.select(range(11))
    for data in train_dataset_small:
        print('----System-----')
        print(data['system'])
        print('------Convo--------')
        for convo in data['conversations']:
            print(convo['from'])
            print(convo['value'])
            print()
            print()
        print('------NEXT----------')
        print('------NEXT----------')
        print('------NEXT----------')


class LogGenerationsCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def on_evaluate(self, args, state, control, **kwargs):
        trainer = kwargs["model"]
        tokenizer = self.tokenizer

        inputs = torch.tensor([kwargs["eval_dataloader"].dataset[0]["input_ids"]]).to(args.device)
        output = trainer.generate(input_ids=inputs, max_new_tokens=100)
        
        prompt = tokenizer.decode(inputs[0], skip_special_tokens=True)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        wandb.log({
            "sample_prompt": wandb.Html(prompt),
            "sample_generation": wandb.Html(response)
        })

def train_and_eval(model, tokenizer, train_dataset, eval_dataset, training_args):

    model.train()
    print('Dataset')
    print(train_dataset[0])
    train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True, collate_fn=default_data_collator)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=default_data_collator)
    for batch in train_dataloader:
        print(batch)
        break
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps,
    )

    step = 0
    for epoch in range(1):
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % training_args.logging_steps == 0:
                wandb.log({"train/loss": loss.item(), "step": step})

            if step % training_args.eval_steps == 0 and step > 0:
                model.eval()
                with torch.no_grad():
                    eval_loss = 0
                    for eval_batch in eval_dataloader:
                        eval_batch = {k: v.to(model.device) for k, v in eval_batch.items()}
                        outputs = model(**eval_batch)
                        eval_loss += outputs.loss.item()
                    avg_eval_loss = eval_loss / len(eval_dataloader)
                    wandb.log({"eval/loss": avg_eval_loss, "step": step})
                model.train()

            step += 1
            if step >= training_args.max_steps:
                return
            
def main():
    

    # --- Setup ---
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    wandb_project = "tableQA-SFT"
    dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k")

    # --- Load tokenizer & model ---
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id, 
    #     device_map="auto", 
    #     torch_dtype=torch.bfloat16
    # )
    loadmodel = LoadModel(pretrained_model=model_id, tune_type='4bit', device='auto')  #4bit #AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
    model = loadmodel.load_model()

    # --- Format each example using chat template ---
    def format_chat(example):
        messages = [{"role": "system", "content": example["system"]}]
        for turn in example["conversations"]:
            if turn["from"] in ["user", "assistant"]:
                messages.append({"role": turn["from"], "content": turn["value"]})
        
        # Infer the correct flags
        last_role = messages[-1]["role"]
        if last_role == "user":
            add_generation_prompt = True
            continue_final_message = False
        elif last_role == "assistant":
            add_generation_prompt = False
            continue_final_message = True
        else:
            raise ValueError(f"Invalid role: {last_role}")
        
        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )
        return {"text": full_prompt}

    # Apply formatting
    dataset = dataset.map(format_chat)
    dataset = dataset["train"]
    dataset = dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    train_dataset = train_dataset.select(range(11))
    

    # --- Define Training Config ---
    training_args = SFTConfig(
        output_dir="./sft_output",
        logging_dir="./logs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_seq_length=2048,
        learning_rate=2e-5,
        max_steps=1000,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        evaluation_strategy="steps",
        report_to="wandb",
        run_name="llama3-bespoke-sft",
        push_to_hub=False,
        bf16=True,
    )

    # --- Init WandB ---
    wandb.init(project=wandb_project, name="llama3-bespoke-sft")
    train_and_eval(model, tokenizer, train_dataset, eval_dataset, training_args)
    # # --- Trainer ---
    # trainer = SFTTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    # )


    # trainer.add_callback(LogGenerationsCallback(tokenizer))



    # --- Train ---
    trainer.train()

if __name__=="__main__":
    main()