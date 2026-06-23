import os
import sys
import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate adapters on custom hybrid dataset.")
    parser.add_argument("--base_model_id", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--math_adapter", type=str, required=True)
    parser.add_argument("--code_adapter", type=str, required=True)
    parser.add_argument("--active_adapter", type=str, choices=["math", "code", "merged"], default="merged")
    parser.add_argument("--merge_method", type=str, default="linear", choices=["linear", "ties", "slerp", "dare_ties"])
    parser.add_argument("--math_weight", type=float, default=0.5)
    parser.add_argument("--code_weight", type=float, default=0.5)
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--input_file", type=str, required=True, help="Path to hybrid_tasks.json")
    parser.add_argument("--output_file", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"1. Loading Hybrid Dataset from {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print("2. Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n3. Loading Base Model ({args.base_model_id})...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, quantization_config=quant_config, device_map={"": 0}, low_cpu_mem_usage=True
    )

    print("\n4. Injecting Multi-Adapter Framework...")
    model = PeftModel.from_pretrained(base_model, args.math_adapter, adapter_name="math")
    model.load_adapter(args.code_adapter, adapter_name="code")

    if args.active_adapter == "merged":
        print(f"\n>>> Fusing Adapters via {args.merge_method.upper()} (Math: {args.math_weight}, Code: {args.code_weight})")
        model.add_weighted_adapter(
            adapters=["math", "code"],
            weights=[args.math_weight, args.code_weight],
            adapter_name="merged",
            combination_type=args.merge_method,
            density=args.density,
        )
        model.set_adapter("merged")
    else:
        print(f"\n>>> Activating Solo Adapter: '{args.active_adapter}'")
        model.set_adapter(args.active_adapter)

    model.eval()

    print(f"\n--- Starting Inference ({len(test_data)} Samples) ---")
    results = []

    for item in tqdm(test_data, desc=f"Evaluating {args.active_adapter.upper()}"):
        query = str(item["query"]).strip()
        prompt = f"Question: {query}\nAnswer:\n"
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
            "task_id": item.get("id", "unknown"),
            "domain": "hybrid",
            "query": query,
            "ground_truth": str(item.get("ground_truth", "")).strip(),
            "raw_output": response.strip(),
        })

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()