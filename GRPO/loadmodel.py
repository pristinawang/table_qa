from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from peft import get_peft_model
import torch

class LoadModel:
    def __init__(self, pretrained_model, tune_type, device):
        self.tune_type=tune_type
        self.pretrained_model=pretrained_model
        self.device=device
        #self.peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
        self.peft_config = LoraConfig(task_type='CAUSAL_LM', inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    
    def load_model(self):
        if self.tune_type == "8bit":
            config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf4",
                bnb_8bit_use_double_quant=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.pretrained_model,
                quantization_config=config,
                device_map=self.device
            )
            model.add_adapter(self.peft_config, adapter_name="adapter_lora")

        elif self.tune_type == "4bit":
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.pretrained_model,
                quantization_config=config,
                device_map=self.device
            )
            model.add_adapter(self.peft_config, adapter_name="adapter_lora")

        elif self.tune_type == 'lora':
            peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
            model = AutoModelForCausalLM.from_pretrained(
                self.pretrained_model,
                device_map=self.device
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        elif self.tune_type == "none":
            # model_kwargs = {"torch_dtype": torch.float16, "device_map": self.device} 
            model = AutoModelForCausalLM.from_pretrained(self.pretrained_model, device_map=self.device)
        else:   
            raise ValueError("value entered for tune_type isn't defined")

        return model
