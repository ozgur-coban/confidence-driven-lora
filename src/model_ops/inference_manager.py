import time
from typing import Optional, Union, Dict

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
)

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
        self.tokenizer.padding_side = "left"

        # FIX 1: Set Llama 3 dedicated pad token to avoid pad == eos batching deadlocks
        if self.tokenizer.pad_token_id is None:
            pad_id = self.tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
            if pad_id is not None and isinstance(pad_id, int):
                self.tokenizer.pad_token_id = pad_id
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Fix EOS token handling for Llama 3.2
        eos_tokens = [self.tokenizer.eos_token_id]
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and isinstance(eot_id, int):
            eos_tokens.append(eot_id)
        self.eos_token_id = eos_tokens

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Loading base model in 16-bit ({dtype}) with eager attention...")
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, 
            torch_dtype=dtype,
            attn_implementation="eager",
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

    def _resolve_kwargs(self, domain: str, runtime_kwargs: dict) -> dict:
        """Extracts kwargs from DOMAIN_REGISTRY and guarantees max_new_tokens is set."""
        config = self.registry.get(domain, {})
        gen_kwargs = config.get("kwargs", {}).copy()
        
        # Override with any explicit kwargs passed at runtime
        gen_kwargs.update(runtime_kwargs)

        # FIX 2: Hard fallback if max_new_tokens was missing due to key mismatch or registry omission
        if "max_new_tokens" not in gen_kwargs and "max_length" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 512

        return gen_kwargs

    def generate(
        self,
        user_prompt: str,
        domain: str,
        stream: bool = True,
        bypass_lora: bool = False,
        **kwargs,
    ):
        gen_kwargs = self._resolve_kwargs(domain, kwargs)
        prompt_str = f"Question: {user_prompt}\nAnswer:\n"
        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)

        streamer = (
            TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            if stream
            else None
        )

        switch_ms = 0.0
        t_gen_start = time.perf_counter()

        if bypass_lora or domain == "general" or not self.adapters_loaded:
            if self.adapters_loaded:
                t0 = time.perf_counter()
                with self.model.disable_adapter():
                    switch_ms = (time.perf_counter() - t0) * 1000
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.eos_token_id,
                            streamer=streamer,
                            **gen_kwargs,
                        )
            else:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.eos_token_id,
                        streamer=streamer,
                        **gen_kwargs,
                    )
        else:
            t0 = time.perf_counter()
            self.model.set_adapter(domain)
            switch_ms = (time.perf_counter() - t0) * 1000
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    streamer=streamer,
                    **gen_kwargs,
                )
                
        t_gen_end = time.perf_counter()
        gen_ms = (t_gen_end - t_gen_start) * 1000

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        num_tokens = len(generated_tokens)
        
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return {
            "response": response,
            "num_tokens": num_tokens,
            "switch_ms": switch_ms,
            "generation_ms": gen_ms,
            "tps": (num_tokens / (gen_ms / 1000)) if gen_ms > 0 else 0.0
        }

    def generate_batch(
        self,
        user_prompts: list,
        domain: str,
        bypass_lora: bool = False,
        **kwargs,
    ):
        gen_kwargs = self._resolve_kwargs(domain, kwargs)
        
        formatted_prompts = [f"Question: {p}\nAnswer:\n" for p in user_prompts]
        inputs = self.tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        switch_ms = 0.0
        t_gen_start = time.perf_counter()

        if bypass_lora or domain == "general" or not self.adapters_loaded:
            if self.adapters_loaded:
                t0 = time.perf_counter()
                with self.model.disable_adapter():
                    switch_ms = (time.perf_counter() - t0) * 1000
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.eos_token_id,
                            **gen_kwargs,
                        )
            else:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.eos_token_id,
                        **gen_kwargs,
                    )
        else:
            t0 = time.perf_counter()
            self.model.set_adapter(domain)
            switch_ms = (time.perf_counter() - t0) * 1000
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    **gen_kwargs,
                )
                
        t_gen_end = time.perf_counter()
        gen_ms = (t_gen_end - t_gen_start) * 1000

        input_length = inputs["input_ids"].shape[1]
        results = []
        
        for i, out_ids in enumerate(outputs):
            gen_ids = out_ids[input_length:]
            num_tokens = len(gen_ids)
            response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            
            amortized_gen_ms = gen_ms / len(user_prompts)
            results.append({
                "response": response,
                "num_tokens": num_tokens,
                "switch_ms": switch_ms / len(user_prompts),
                "generation_ms": amortized_gen_ms,
                "tps": (num_tokens / (amortized_gen_ms / 1000)) if amortized_gen_ms > 0 else 0.0
            })
            
        return results