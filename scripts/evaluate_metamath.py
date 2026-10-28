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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained LoRA adapter on MetaMathQA."
    )
    parser.add_argument(
        "--base_model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to adapter. If None, runs base model.",
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
        help="Number of test samples to evaluate.",
    )
    parser.add_argument(
        "--train_data_size",
        type=int,
        default=15000,
        help="Matches training size to prevent leakage.",
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Max token length for VRAM safety."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="artifacts/generated_results/metamath_eval_results.json",
    )
    return parser.parse_args()


def clean_math_string(s: str) -> str:
    """Normalizes math outputs: unwraps \boxed{}, cleans $, and removes fluff."""
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
            s = nums[-1]

    if re.match(r"^-?[\d,]+(?:\.\d+)?$", s):
        s = s.replace(",", "")
    return s.strip()


def extract_model_answer(text: str) -> str:
    """Extracts final answer with fallback logic for non-compliant model outputs."""
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


def extract_ground_truth(text: str) -> str:
    """Extracts answer from MetaMathQA ground truth format."""
    if "The answer is:" in text:
        ans = text.split("The answer is:")[-1].strip()
        return clean_math_string(ans)
    return "Extraction Failed"


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.domain not in DOMAIN_REGISTRY:
        raise ValueError(f"Domain '{args.domain}' not in registry.")
    domain_cfg = DOMAIN_REGISTRY[args.domain]

    print(f"1. Loading Tokenizer and Data (Domain: {args.domain})...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    dataset = load_dataset("meta-math/MetaMathQA", split="train")

    # --- Dynamic Leakage Prevention ---
    shuffled_dataset = dataset.shuffle(seed=42)
    rows_touched_by_training = args.train_data_size * 5
    unseen_data = shuffled_dataset.select(
        range(rows_touched_by_training, len(shuffled_dataset))
    )

    # --- VRAM Protection via Strict Token Filtering ---
    print(f"Filtering unseen dataset for max_seq_length <= {args.max_length}...")

    def is_within_length(example):
        messages = [
            {"role": "system", "content": domain_cfg["system"]},
            {"role": "user", "content": example["query"]},
            {"role": "assistant", "content": str(example["response"])},
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        return len(tokenizer.encode(full_text)) <= args.max_length

    filtered_unseen_data = unseen_data.filter(is_within_length, num_proc=4)
    num_to_evaluate = (
        len(filtered_unseen_data)
        if args.num_samples == -1
        else min(args.num_samples, len(filtered_unseen_data))
    )
    test_data = filtered_unseen_data.select(range(num_to_evaluate))

    print(f"2. Loading Base Model ({args.base_model_id})...")
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
    results = []
    correct_count = 0
    failed_extraction_count = 0

    for i, example in enumerate(tqdm(test_data, desc="Evaluating")):
        messages = [
            {"role": "system", "content": domain_cfg["system"]},
            {"role": "user", "content": example["query"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **domain_cfg[
                    "kwargs"
                ],  # Dynamically injects max_new_tokens, temp, etc.
                pad_token_id=tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        extracted_ans = extract_model_answer(response)
        ground_truth = extract_ground_truth(example["response"])

        is_correct = extracted_ans == ground_truth
        if extracted_ans == "Extraction Failed":
            failed_extraction_count += 1
        elif is_correct:
            correct_count += 1

        results.append(
            {
                "id": i,
                "query": example["query"],
                "ground_truth": ground_truth,
                "predicted_answer": extracted_ans,
                "raw_output": response,
                "is_correct": is_correct,
            }
        )

    # Print Final Statistics
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
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
