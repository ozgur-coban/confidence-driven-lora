import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


class LoraManager:
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.model_id = model_id
        # Automatically detect GPU
        self.device_map = "auto" if torch.cuda.is_available() else "cpu"

    def load_base_model(self):
        """Loads the base Llama 3 model in 4-bit quantization for Colab T4."""
        print(f"Loading base model in 4-bit: {self.model_id}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # CHANGED FROM bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,  # Critical for memory management
            device_map=self.device_map,
        )
        model = prepare_model_for_kbit_training(model)
        # Disable cache during training to save memory
        model.config.use_cache = False
        return model

    def get_peft_setup(
        self, model, r: int = 16, alpha: int = 32, target_modules: list = None
    ):
        """Wraps the base model with the LoRA adapter configuration."""
        print("Injecting LoRA adapters...")
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            lora_dropout=0.0,
        )

        peft_model = get_peft_model(model, config)

        # Print a summary of how many parameters are actually trainable
        peft_model.print_trainable_parameters()

        return peft_model
