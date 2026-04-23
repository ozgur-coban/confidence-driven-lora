from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch


class LoraManager:
    def __init__(self, model_id: str, max_length: int = 512):
        self.model_id = model_id
        self.max_length = max_length

    def load_base_model(self):
        """Loads the model and tokenizer using Unsloth's optimized backend."""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_id,
            max_seq_length=self.max_length,
            dtype=None,  # Auto-detects float16 for Colab T4
            load_in_4bit=True,
        )

        # CRITICAL FIX FOR BASE MODELS:
        # The base Llama 3.1 model does not know how to format chat conversations.
        # This forcefully injects the Instruct chat template into the tokenizer
        # so your Preprocessors don't crash.
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

        return model, tokenizer

    def get_peft_setup(self, model, r: int, alpha: int, target_modules: list):
        """Wraps the model in Unsloth's highly optimized LoRA architecture."""
        peft_model = FastLanguageModel.get_peft_model(
            model,
            r=r,
            target_modules=target_modules,
            lora_alpha=alpha,
            lora_dropout=0,  # Unsloth requires 0 dropout for optimized training
            bias="none",
            use_gradient_checkpointing="unsloth",  # Massive VRAM savings
            random_state=42,
            use_rslora=False,
        )
        return peft_model
