import torch
import argparse
import json
import re
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained LoRA adapter or base model on MedQA USMLE.")
    parser.add_argument("--base_model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to the trained MedQA LoRA adapter. If omitted, runs base model.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples to evaluate. Set to -1 for the full test set.")
    parser.add_argument("--output_file", type=str, default="artifacts/generated_results/medqa_eval_results.json")
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

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. Loading MedQA Test Set...")
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    
    # Handle -1 for full dataset
    num_to_evaluate = len(dataset) if args.num_samples == -1 else min(args.num_samples, len(dataset))
    dataset = dataset.select(range(num_to_evaluate))

    print(f"2. Loading Base Model: {args.base_model_id}")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, 
        quantization_config=quant_config, 
        device_map="auto"
    )
    
    if args.adapter_path:
        print(f"Injecting Adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    else:
        print("No adapter path provided. Running base model only.")
        model = base_model
        
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)

    print(f"\n--- Evaluating {len(dataset)} MedQA Questions ---")
    results = []
    correct_count = 0

    for i, item in enumerate(tqdm(dataset)):
        question = str(item.get("question", "")).strip()
        options = item.get("options", {})
        ground_truth_idx = str(item.get("answer_idx", "")).strip()
        
        # 1. Recreate the Exact Training Prompt
        options_list = []
        for letter in ["A", "B", "C", "D"]:
            if letter in options:
                options_list.append(f"{letter}) {options[letter]}")
        options_str = "\n".join(options_list)
        
        input_text = f"{question}\n\nOptions:\n{options_str}"
        
        messages = [
            {
                "role": "system", 
                "content": "You are a medical expert taking the USMLE board exam. Read the clinical vignette and options carefully. Output ONLY the single letter of the correct answer (A, B, C, or D). Do not provide any explanation."
            },
            {
                "role": "user", 
                "content": input_text
            }
        ]
        
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)

        # 2. Run Inference
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=10, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # 3. Extract and Score
        predicted_letter = extract_answer_letter(generated_text)
        is_correct = (predicted_letter == ground_truth_idx)
        
        if is_correct:
            correct_count += 1

        results.append({
            "id": i,
            "question": question,
            "ground_truth": ground_truth_idx,
            "predicted_letter": predicted_letter,
            "raw_output": generated_text,
            "is_correct": is_correct
        })

    # 4. Calculate Final Metrics
    accuracy = (correct_count / len(dataset)) * 100

    print("\n" + "="*50)
    print("FINAL MEDQA (USMLE) EVALUATION METRICS")
    print("="*50)
    print(f"Total Questions Evaluated: {len(dataset)}")
    print(f"Correct Answers:           {correct_count}")
    print(f"Diagnostic Accuracy:       {accuracy:.2f}%")
    print("="*50)

    # Save detailed logs
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nDetailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()