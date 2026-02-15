import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

class LoraManager:
    def __init__(self, model_id: str = "NousResearch/Meta-Llama-3-8B-Instruct"):
        self.model_id = model_id
        # Automatically detect GPU
        self.device_map = "auto" if torch.cuda.is_available() else "cpu"

    def load_base_model(self):
        """Loads the frozen base Llama 3 model in bfloat16 precision."""
        print(f"Loading base model: {self.model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16, # Critical for memory management
            device_map=self.device_map
        )
        # Disable cache during training to save memory
        model.config.use_cache = False 
        return model

    def get_peft_setup(self, model, r: int = 16, alpha: int = 32):
        """Wraps the base model with the LoRA adapter configuration."""
        print("Injecting LoRA adapters...")
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            lora_dropout=0.0
        )
        
        peft_model = get_peft_model(model, config)
        
        # Print a summary of how many parameters are actually trainable
        peft_model.print_trainable_parameters()
        
        return peft_model