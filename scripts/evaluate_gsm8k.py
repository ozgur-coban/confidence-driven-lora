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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained base-model LoRA adapter on GSM8K."
    )
    # Updated to Llama 3.2 3B Base
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
        "--use_few_shot",
        action="store_true",
        help="Enable few-shot examples.",
    )
    return parser.parse_args()


def clean_math_string(s: str) -> str:
    """Normalizes math outputs: safely unwraps \\boxed{}, strips $, cleans commas."""
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


def build_static_few_shot(current_query: str) -> str:
    """Provides a strict Base Model autocomplete few-shot context, including GSM8K calculator syntax."""
    example = (
        "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n"
        "Answer:\nNatalia sold 48 / 2 = <<48/2=24>>24 clips in May.\n"
        "Natalia sold 48 + 24 = <<48+24=72>>72 clips altogether in April and May.\n"
        "#### 72\n\n"
    )
    return f"{example}Question: {current_query}\n"


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"1. Loading Tokenizer and Data for domain {args.domain}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure we load 'main' split for base model continuous reasoning
    raw_test_set = load_dataset("openai/gsm8k", "main", split="test")

    print(f"Filtering test set for strict max_seq_length <= {args.max_length}...")

    def is_within_length(example):
        q = example["question"]
        if args.use_few_shot:
            q = build_static_few_shot(q)
        # Simulate base model layout
        full_text = f"Question: {q}\nAnswer:\n{example['answer']}{tokenizer.eos_token}"
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

    print(f"\n--- Starting Evaluation on {len(test_data)} Samples ---")
    print(f"Few-Shot Mode: {'ENABLED' if args.use_few_shot else 'DISABLED'}")

    results = []
    correct_count = 0
    failed_extraction_count = 0

    for i, example in enumerate(tqdm(test_data, desc="Evaluating")):
        q = example["question"]

        if args.use_few_shot:
            q = build_static_few_shot(q)
            prompt = f"{q}Answer:\n"
        else:
            prompt = f"Question: {q}\nAnswer:\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512, # Bound generation so base model doesn't loop forever
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False, # Greedy decoding is mathematically most accurate
            )

        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        extracted_ans = extract_model_answer(response)
        ground_truth = extract_model_answer(example["answer"])

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

    accuracy = (correct_count / len(test_data)) * 100
    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRICS")
    print("=" * 50)
    print(f"Total Samples Tested:  {len(test_data)}")
    print(f"Correct Answers:       {correct_count}")
    print(f"Failed Extractions:    {failed_extraction_count}")
    print(f"Overall Accuracy:      {accuracy:.2f}%")
    print("=" * 50)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nDetailed results saved to {args.output_file}")


if __name__ == "__main__":
    main()