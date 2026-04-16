from unsloth import FastLanguageModel

class LoraManager:
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3.1-8B", max_seq_length: int = 2048):
        self.model_id = model_id
        self.max_seq_length = max_seq_length

    def load_model_and_tokenizer(self):
        """Loads both the model and tokenizer using Unsloth's optimized backend in 4-bit."""
        print(f"Loading Unsloth base model and tokenizer: {self.model_id}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_id,
            max_seq_length=self.max_seq_length,
            dtype=None, # Auto-detects fp16 vs bf16 based on your GPU
            load_in_4bit=True
        )
        return model, tokenizer

    def get_peft_setup(self, model, r: int = 16, alpha: int = 32, target_modules: list = None):
        """Injects Unsloth's highly optimized LoRA matrices."""
        print("Injecting Unsloth LoRA adapters...")
        
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        peft_model = FastLanguageModel.get_peft_model(
            model,
            r=r,
            target_modules=target_modules,
            lora_alpha=alpha,
            lora_dropout=0, # Unsloth specifically requires 0 dropout for maximum speed
            bias="none",
            use_gradient_checkpointing="unsloth", # Unsloth's custom memory saving technique
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        peft_model.print_trainable_parameters()
        return peft_model