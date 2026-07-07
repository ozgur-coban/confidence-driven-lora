import torch
from typing import Union, Dict, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)
from peft import PeftModel
from config.domain_registry import DOMAIN_REGISTRY


class InferenceManager:
    model: Union[AutoModelForCausalLM, PeftModel]

    def __init__(
        self,
        base_model_id: str = "meta-llama/Llama-3.2-3B",
        adapter_paths: Optional[Dict[str, str]] = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.registry = DOMAIN_REGISTRY

        print(f"Loading tokenizer for {base_model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, clean_up_tokenization_spaces=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading 4-bit base model...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, 
            quantization_config=quant_config, 
            device_map={"": 0}
        )

        self.model = self.base_model
        self.adapters_loaded = False

        if adapter_paths and len(adapter_paths) > 0:
            print("Loading LoRA adapters into VRAM...")
            domain_names = list(adapter_paths.keys())

            first_domain = domain_names[0]
            self.model = PeftModel.from_pretrained(
                self.base_model, adapter_paths[first_domain], adapter_name=first_domain
            )

            for domain in domain_names[1:]:
                self.model.load_adapter(adapter_paths[domain], adapter_name=domain)

            self.adapters_loaded = True
            print(f"Adapters loaded: {domain_names}")

    def generate(
        self,
        user_prompt: str,
        domain: str,
        stream: bool = True,
        bypass_lora: bool = False,
        **kwargs,
    ) -> str:
        
        if domain not in self.registry:
            domain = "general"

        config = self.registry.get(domain, {})
        
        prompt_str = f"Question: {user_prompt}\nAnswer:\n"
        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)

        streamer = (
            TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            if stream
            else None
        )

        # ---------------------------------------------------------
        # ROUTING EXECUTION LOGIC
        # ---------------------------------------------------------
        if bypass_lora or domain == "general" or not self.adapters_loaded:
            if self.adapters_loaded:
                with self.model.disable_adapter():
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            **config.get("kwargs", {}),
                            pad_token_id=self.tokenizer.eos_token_id,
                            streamer=streamer,
                            **kwargs,
                        )
            else:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **config.get("kwargs", {}),
                        pad_token_id=self.tokenizer.eos_token_id,
                        streamer=streamer,
                        **kwargs,
                    )
        else:
            self.model.set_adapter(domain)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **config.get("kwargs", {}),
                    pad_token_id=self.tokenizer.eos_token_id,
                    streamer=streamer,
                    **kwargs,
                )

        input_length = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        )

        return response