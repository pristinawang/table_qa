import os
import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

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
    def on_evaluate(self, args, state, control, **kwargs):
        trainer = kwargs["model"]
        tokenizer = kwargs["tokenizer"]

        inputs = torch.tensor([kwargs["eval_dataloader"].dataset[0]["input_ids"]]).to(args.device)
        output = trainer.generate(input_ids=inputs, max_new_tokens=100)
        
        prompt = tokenizer.decode(inputs[0], skip_special_tokens=True)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        wandb.log({
            "sample_prompt": wandb.Html(prompt),
            "sample_generation": wandb.Html(response)
        })
        
def main():
    

    # --- Setup ---
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    wandb_project = "tableQA-SFT"
    dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k")

    # --- Load tokenizer & model ---
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )

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

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )



    trainer.add_callback(LogGenerationsCallback())

    # --- Train ---
    trainer.train()

if __name__=="__main__":
    main()