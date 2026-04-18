import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    parser = argparse.ArgumentParser(description="Evaluate a trained LoRA adapter or base model on MedQA USMLE.")
    parser.add_argument("--base_model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to the trained adapter. If omitted, runs base model.")
    parser.add_argument("--domain", type=str, default="medical", choices=["math", "code", "medical", "general"])
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples to evaluate. Set to -1 for full set.")
    parser.add_argument("--max_length", type=int, default=512, help="Max token length to prevent VRAM OOM.")
    parser.add_argument("--output_file", type=str, default="artifacts/generated_results/medqa_eval_results.json")
    parser.add_argument("--train_data_size", type=int, default=7500, help="Matches training size for pipeline compatibility.")
    return parser.parse_args()

def extract_answer_letter(text: str) -> str:
    """
    Extracts the first occurrence of A, B, C, or D from the model's output.
    Handles edge cases like 'A)', 'A.', or 'The answer is A'.
    """
    match = re.search(r'\b([A-D])\b', text.strip())
    if match:
        return match.group(1)
    # Fallback: just return the first uppercase character if regex fails
    for char in text.strip():
        if char in "ABCD":
            return char
    return "UNKNOWN"

def format_medqa_input(item: dict) -> str:
    """Formats the MedQA question and options into a standard string."""
    question = str(item.get("question", "")).strip()
    options = item.get("options", {})
    
    options_list = []
    for letter in ["A", "B", "C", "D"]:
        if letter in options:
            options_list.append(f"{letter}) {options[letter]}")
    options_str = "\n".join(options_list)
    
    return f"{question}\n\nOptions:\n{options_str}"

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.domain not in DOMAIN_REGISTRY:
        raise ValueError(f"Domain '{args.domain}' not found in DOMAIN_REGISTRY.")
    domain_cfg = DOMAIN_REGISTRY[args.domain]

    print(f"1. Loading Tokenizer and Data for domain: {args.domain}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    raw_test_set = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")

    print(f"Filtering test set for strict max_seq_length <= {args.max_length}...")

    def is_within_length(example):
        input_text = format_medqa_input(example)
        messages = [
            {"role": "system", "content":  "You are a medical expert taking the USMLE board exam. Read the clinical vignette and options carefully. Output ONLY the single letter of the correct answer (A, B, C, or D). Do not provide any explanation."},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": str(example.get("answer_idx", "")).strip()}
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        return len(tokenizer.encode(full_text)) <= args.max_length

    filtered_test_set = raw_test_set.filter(is_within_length, num_proc=4)

    num_to_evaluate = len(filtered_test_set) if args.num_samples == -1 else min(args.num_samples, len(filtered_test_set))
    test_data = filtered_test_set.shuffle(seed=42).select(range(num_to_evaluate))
    print(f"Evaluation dataset ready: {len(test_data)} samples.")

    print(f"2. Loading Base Model ({args.base_model_id}) in 4-bit...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        quantization_config=quant_config,
        device_map={"": 0},
        low_cpu_mem_usage=True 
    )

    if args.adapter_path:
        print(f"3. Injecting LoRA Adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    else:
        print("3. No adapter path provided. Bypassing adapter: Evaluating Base Model only...")
        model = base_model
    
    model.eval()

    print(f"\n--- Starting Evaluation on {len(test_data)} Samples ---")
    
    results = []
    correct_count = 0
    failed_extraction_count = 0

    for i, example in enumerate(tqdm(test_data, desc="Evaluating")):
        input_text = format_medqa_input(example)
        ground_truth_idx = str(example.get("answer_idx", "")).strip()

        messages = [
            {"role": "system", "content": "You are a medical expert taking the USMLE board exam. Read the clinical vignette and options carefully. Output ONLY the single letter of the correct answer (A, B, C, or D). Do not provide any explanation."},
            {"role": "user", "content": input_text}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Merge kwargs but FORCE max_new_tokens to 10 for evaluation speed
        gen_kwargs = domain_cfg["kwargs"].copy()
        gen_kwargs["max_new_tokens"] = 10 

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_kwargs,
                pad_token_id=tokenizer.eos_token_id
            )
        
        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        predicted_letter = extract_answer_letter(response)
        is_correct = (predicted_letter == ground_truth_idx)
        
        if predicted_letter == "UNKNOWN":
            failed_extraction_count += 1
        elif is_correct:
            correct_count += 1

        results.append({
            "id": i,
            "question": input_text,
            "ground_truth": ground_truth_idx,
            "predicted_letter": predicted_letter,
            "raw_output": response,
            "is_correct": is_correct
        })

    # Final Statistics
    accuracy = (correct_count / len(test_data)) * 100
    print("\n" + "="*50)
    print("FINAL MEDQA (USMLE) EVALUATION METRICS")
    print("="*50)
    print(f"Total Samples Tested:  {len(test_data)}")
    print(f"Correct Answers:       {correct_count}")
    print(f"Failed Extractions:    {failed_extraction_count}")
    print(f"Diagnostic Accuracy:   {accuracy:.2f}%")
    print("="*50)

    # Save detailed logs
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nDetailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()