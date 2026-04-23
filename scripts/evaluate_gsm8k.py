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
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from config.domain_registry import DOMAIN_REGISTRY
from config.few_shot_prefixes import FEW_SHOT_PREFIXES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained LoRA adapter on GSM8K."
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
        default="math",
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
        default=1500,
        help="Max token length to prevent VRAM OOM.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="artifacts/generated_results/gsm8k_eval_results.json",
    )
    parser.add_argument(
        "--train_data_size",
        type=int,
        default=15000,
        help="Matches training size for pipeline compatibility.",
    )
    # Added the proper boolean toggle for few-shot
    parser.add_argument(
        "--use_few_shot",
        action="store_true",
        help="Enable few-shot examples from the dictionary.",
    )
    return parser.parse_args()


def clean_math_string(s: str) -> str:
    """Normalizes math outputs: safely unwraps \boxed{}, strips $, cleans commas, and parses sentences."""
    s = s.strip()
    s = re.sub(r"^#+", "", s)
    s = re.sub(r"#+$", "", s)
    s = s.strip()

    if "\\boxed{" in s:
        start = s.find("\\boxed{") + 7
        depth = 1
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
            if depth == 0:
                s = s[start:i]
                break

    s = s.strip("$ \n\t.")
    s = re.sub(r"^(The )?(final )?answer is\s*:?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^Therefore,?\s*(the )?.*?\s+is\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^So,?\s*(the )?.*?\s+is\s*", "", s, flags=re.IGNORECASE)
    s = s.strip("$ \n\t.")

    if len(s) > 10 and not any(
        op in s for op in ["+", "-", "*", "/", "\\", "^", "=", "x", "y", "z"]
    ):
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", s)
        if nums:
            s = s.append(nums[-1])

    if re.match(r"^-?[\d,]+(?:\.\d+)?$", s):
        s = s.replace(",", "")

    return s.strip()


def extract_model_answer(text: str) -> str:
    """Robust extraction handling ####, common phrases, and raw number fallbacks."""
    ans = "Extraction Failed"

    if "####" in text:
        parts = text.split("####")
        ans = parts[1] if len(parts) > 1 and parts[1].strip() else parts[-1]
    elif "The final answer is:" in text:
        ans = text.split("The final answer is:")[-1]
    elif "The answer is:" in text:
        ans = text.split("The answer is:")[-1]
    else:
        numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
        if numbers:
            ans = numbers[-1]

    if ans == "Extraction Failed":
        return ans
    return clean_math_string(ans)


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

    print(f"1. Loading Tokenizer and Data for domain {args.domain}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    raw_test_set = load_dataset("openai/gsm8k", "socratic", split="test")

    print(f"Filtering test set for strict max_seq_length <= {args.max_length}...")

    def is_within_length(example):
        user_content = example["question"]
        # Apply few-shot during the token counting phase if the flag is active
        if args.use_few_shot:
            user_content = build_static_few_shot(user_content, args.domain)

        messages = [
            {"role": "system", "content": domain_cfg["system"]},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": str(example["answer"])},
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

    print(f"\n--- Starting Evaluation on {len(test_data)} Samples ---")
    print(f"Few-Shot Mode: {'ENABLED' if args.use_few_shot else 'DISABLED'}")

    results = []
    correct_count = 0
    failed_extraction_count = 0

    for i, example in enumerate(tqdm(test_data, desc="Evaluating")):
        user_content = example["question"]

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

        extracted_ans = extract_model_answer(response)
        ground_truth = extract_model_answer(
            example["answer"]
        )  # Consistent cleaning for both

        is_correct = extracted_ans == ground_truth
        if extracted_ans == "Extraction Failed":
            failed_extraction_count += 1
        elif is_correct:
            correct_count += 1

        results.append(
            {
                "id": i,
                "query": example["question"],
                "ground_truth": ground_truth,
                "predicted_answer": extracted_ans,
                "raw_output": response,
                "is_correct": is_correct,
            }
        )

    # Final Statistics
    accuracy = (correct_count / len(test_data)) * 100
    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRICS")
    print("=" * 50)
    print(f"Total Samples Tested:  {len(test_data)}")
    print(f"Correct Answers:       {correct_count}")
    print(f"Failed Extractions:    {failed_extraction_count}")
    print(f"Overall Accuracy:      {accuracy:.2f}%")
    print("=" * 50)

    # Save detailed logs
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nDetailed results saved to {args.output_file}")


if __name__ == "__main__":
    main()
