import os
import sys
import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate merged LoRA adapters on a mixed Math/Code dataset.")
    parser.add_argument("--base_model_id", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--math_adapter", type=str, required=True, help="Path to Math adapter")
    parser.add_argument("--code_adapter", type=str, required=True, help="Path to Code adapter")
    parser.add_argument("--active_adapter", type=str, choices=["math", "code", "merged"], default="merged", help="Which adapter to activate for this run.")
    parser.add_argument("--merge_method", type=str, default="linear", choices=["linear", "ties", "slerp", "dare_ties"])
    parser.add_argument("--math_weight", type=float, default=0.5)
    parser.add_argument("--code_weight", type=float, default=0.5)
    parser.add_argument("--density", type=float, default=0.2, help="Density for TIES/DARE merging")
    parser.add_argument("--samples_per_domain", type=int, default=50, help="Number of samples from EACH domain (e.g., 50 = 100 total)")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_file", type=str, default="artifacts/generated_results/mixed_eval_results.json")
    return parser.parse_args()

def build_mixed_dataset(tokenizer, samples_per_domain, max_length):
    """Creates a unified 50/50 dataset of GSM8K and MBPP, filtered by zero-shot token length."""
    print(f"Building mixed dataset ({samples_per_domain} Math + {samples_per_domain} Code)...")
    
    mixed_data = []

    # 1. Process GSM8K
    gsm8k_raw = load_dataset("openai/gsm8k", "main", split="test")
    def is_valid_math(example):
        text = f"Question: {example['question']}\nAnswer:\n{example['answer']}{tokenizer.eos_token}"
        return len(tokenizer.encode(text)) <= max_length
    
    gsm8k_filtered = gsm8k_raw.filter(is_valid_math, num_proc=4).shuffle(seed=42).select(range(samples_per_domain))
    for i, row in enumerate(gsm8k_filtered):
        mixed_data.append({
            "task_id": f"math_{i}",
            "domain": "math",
            "query": str(row["question"]).strip(),
            "ground_truth": str(row["answer"]).strip()
        })

    # 2. Process MBPP
    mbpp_raw = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    def is_valid_code(example):
        text = f"Question: {example['prompt']}\nAnswer:\n{example['code']}{tokenizer.eos_token}"
        return len(tokenizer.encode(text)) <= max_length

    mbpp_filtered = mbpp_raw.filter(is_valid_code, num_proc=4).shuffle(seed=42).select(range(samples_per_domain))
    for i, row in enumerate(mbpp_filtered):
        mixed_data.append({
            "task_id": f"code_{row.get('task_id', i)}",
            "domain": "code",
            "query": str(row["prompt"]).strip(),
            "ground_truth": str(row["code"]).strip()
        })

    return mixed_data

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_data = build_mixed_dataset(tokenizer, args.samples_per_domain, args.max_length)
    print(f"Unified dataset ready: {len(test_data)} total samples.")

    print(f"\n2. Loading Base Model ({args.base_model_id})...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, quantization_config=quant_config, device_map={"": 0}, low_cpu_mem_usage=True
    )

    print("\n3. Injecting Multi-Adapter Framework...")
    # Load Math as the primary adapter, but don't activate it exclusively yet
    model = PeftModel.from_pretrained(base_model, args.math_adapter, adapter_name="math")
    # Load Code as a secondary adapter into the same model mapping
    model.load_adapter(args.code_adapter, adapter_name="code")

    if args.active_adapter == "merged":
        print(f"\n>>> Fusing Adapters via {args.merge_method.upper()} (Math: {args.math_weight}, Code: {args.code_weight})")
        model.add_weighted_adapter(
            adapters=["math", "code"],
            weights=[args.math_weight, args.code_weight],
            adapter_name="merged",
            combination_type=args.merge_method,
            density=args.density, # Only used if method is ties/dare
        )
        model.set_adapter("merged")
    else:
        print(f"\n>>> Activating Solo Adapter: '{args.active_adapter}'")
        model.set_adapter(args.active_adapter)

    model.eval()

    print(f"\n--- Starting Inference ({len(test_data)} Samples) ---")
    results = []

    for item in tqdm(test_data, desc=f"Evaluating {args.active_adapter.upper()}"):
        prompt = f"Question: {item['query']}\nAnswer:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False, 
            )

        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        results.append({
            "task_id": item["task_id"],
            "domain": item["domain"],
            "query": item["query"],
            "ground_truth": item["ground_truth"],
            "raw_output": response.strip(),
        })

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()