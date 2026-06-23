import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import argparse
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate base-model code responses for the MBPP dataset."
    )
    parser.add_argument(
        "--base_model_id", type=str, default="meta-llama/Llama-3.2-3B"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to the trained adapter. If omitted, runs base model.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="code",
        choices=["math", "code", "medical", "general"],
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of test samples to evaluate. Set to -1 for full set.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max token length to maintain training distribution parity.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="artifacts/generated_results/mbpp_eval_results.json",
    )
    parser.add_argument(
        "--use_few_shot",
        action="store_true",
        help="Enable few-shot examples.",
    )
    return parser.parse_args()


def build_static_few_shot(current_query: str) -> str:
    """Provides a strict Base Model autocomplete few-shot context for Python."""
    example = (
        "Question: Write a python function to add two numbers.\n"
        "Answer:\n"
        "def add_numbers(a, b):\n"
        "    return a + b\n\n"
    )
    return f"{example}Question: {current_query}\n"


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"1. Loading Tokenizer and Data for domain: {args.domain}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_test_set = load_dataset(
        "google-research-datasets/mbpp", "sanitized", split="test"
    )

    print(f"Filtering test set for strict max_seq_length <= {args.max_length}...")

    def is_within_length(example):
        # Count strictly zero-shot tokens to ensure 1-to-1 matching across all test runs
        q = str(example["prompt"]).strip()
        code = str(example["code"]).strip()
        full_text = f"Question: {q}\nAnswer:\n{code}{tokenizer.eos_token}"
        return len(tokenizer.encode(full_text)) <= args.max_length

    filtered_test_set = raw_test_set.filter(is_within_length, num_proc=4)

    num_to_evaluate = (
        len(filtered_test_set)
        if args.num_samples == -1
        else min(args.num_samples, len(filtered_test_set))
    )
    test_data = filtered_test_set.shuffle(seed=42).select(range(num_to_evaluate))
    print(f"Evaluation dataset ready: {len(test_data)} samples.")

    print(f"2. Loading Base Model ({args.base_model_id}) in 4-bit...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        quantization_config=quant_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )

    if args.adapter_path:
        print(f"3. Injecting LoRA Adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    else:
        print("3. No adapter path provided. Evaluating Base Model only...")
        model = base_model

    model.eval()

    print(f"\n--- Starting Generation on {len(test_data)} Samples ---")
    print(f"Few-Shot Mode: {'ENABLED' if args.use_few_shot else 'DISABLED'}")

    results = []

    for i, example in enumerate(tqdm(test_data, desc="Generating Code")):
        user_query = str(example["prompt"]).strip()
        ground_truth_code = str(example["code"]).strip()

        if args.use_few_shot:
            q = build_static_few_shot(user_query)
            prompt = f"{q}Answer:\n"
        else:
            prompt = f"Question: {user_query}\nAnswer:\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512, # Bound generation
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False, # Greedy decoding for exact syntax
            )

        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        results.append(
            {
                "task_id": example.get("task_id", i),
                "query": user_query,
                "ground_truth_code": ground_truth_code,
                "raw_output": response.strip(),
            }
        )

    print("\n" + "=" * 50)
    print("GENERATION COMPLETE")
    print("=" * 50)
    print(f"Total Samples Processed: {len(test_data)}")
    print("Output dumped directly to JSON. No execution-based accuracy calculated.")
    print("=" * 50)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nRaw outputs saved to {args.output_file}")


if __name__ == "__main__":
    main()