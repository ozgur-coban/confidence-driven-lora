"""
Copyright (C) 2026 Özgür ÇOBAN

This file is part of the confidence-driven-lora.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

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
from config.domain_registry import DOMAIN_REGISTRY
from config.few_shot_prefixes import FEW_SHOT_PREFIXES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate responses for the MBPP dataset using a trained LoRA adapter or base model."
    )
    parser.add_argument(
        "--base_model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct"
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
        default=4096,
        help="Max token length to prevent VRAM OOM. Bumped to 4096 for Code Few-Shot.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="artifacts/generated_results/mbpp_eval_results.json",
    )
    parser.add_argument(
        "--use_few_shot",
        action="store_true",
        help="Enable few-shot examples from the dictionary.",
    )
    return parser.parse_args()


def build_static_few_shot(current_query: str, domain: str) -> str:
    """Prepends few-shot examples and adds the strict context boundary."""
    prefix = FEW_SHOT_PREFIXES.get(domain, "")
    if not prefix.strip():
        return current_query
    return (
        f"{prefix}\nNow solve the following problem:\nInput: {current_query}\nAnswer:"
    )


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.domain not in DOMAIN_REGISTRY:
        raise ValueError(f"Domain '{args.domain}' not found in DOMAIN_REGISTRY.")
    domain_cfg = DOMAIN_REGISTRY[args.domain]

    print(f"1. Loading Tokenizer and Data for domain: {args.domain}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)

    # MBPP usually has a standard 'test' split.
    # 1.  define the direct URLs to the data files to bypass the script
    raw_test_set = load_dataset(
        "google-research-datasets/mbpp", "sanitized", split="test"
    )

    print(f"Filtering test set for strict max_seq_length <= {args.max_length}...")

    def is_within_length(example):
        user_content = str(example["prompt"]).strip()

        # Apply few-shot during the token counting phase if the flag is active
        if args.use_few_shot:
            user_content = build_static_few_shot(user_content, args.domain)

        messages = [
            {"role": "system", "content": domain_cfg["system"]},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": str(example["code"]).strip()},
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
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
        print(
            "3. No adapter path provided. Bypassing adapter: Evaluating Base Model only..."
        )
        model = base_model

    model.eval()

    print(f"\n--- Starting Generation on {len(test_data)} Samples ---")
    print(f"Few-Shot Mode: {'ENABLED' if args.use_few_shot else 'DISABLED'}")

    results = []

    for i, example in enumerate(tqdm(test_data, desc="Generating Code")):
        user_query = str(example["prompt"]).strip()
        ground_truth_code = str(example["code"]).strip()

        user_content = user_query

        # Apply the few-shot builder before generation
        if args.use_few_shot:
            user_content = build_static_few_shot(user_content, args.domain)

        messages = [
            {"role": "system", "content": domain_cfg["system"]},
            {"role": "user", "content": user_content},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, **domain_cfg["kwargs"], pad_token_id=tokenizer.eos_token_id
            )

        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        results.append(
            {
                "task_id": example.get("task_id", i),
                "query": user_query,
                "ground_truth_code": ground_truth_code,
                "raw_output": response,
            }
        )

    print("\n" + "=" * 50)
    print("GENERATION COMPLETE")
    print("=" * 50)
    print(f"Total Samples Processed: {len(test_data)}")
    print(
        "Note: Output is dumped directly to JSON. No execution-based accuracy was calculated."
    )
    print("=" * 50)

    # Save detailed logs
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nRaw outputs saved to {args.output_file}")


if __name__ == "__main__":
    main()
