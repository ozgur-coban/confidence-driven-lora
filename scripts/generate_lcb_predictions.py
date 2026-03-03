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
    parser = argparse.ArgumentParser(description="Generate LiveCodeBench predictions using a base model or trained LoRA adapter.")
    parser.add_argument("--base_model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to the trained LoRA adapter. If omitted, runs base model.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of recent, contamination-free LCB samples to evaluate.")
    parser.add_argument("--output_file", type=str, default="artifacts/generated_results/lcb_predictions.json")
    return parser.parse_args()

def extract_code(text: str) -> str:
    """
    Attempts to extract pure code from markdown backticks.
    If the model learned perfectly from CodeFeedback, there won't be backticks,
    so it falls back to grabbing the raw text.
    """
    match = re.search(r'```(?:python)?(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. Loading LiveCodeBench Dataset (Contamination-Free)...")
    try:
        # LCB typically uses 'test' split for evaluations
        dataset = load_dataset("sam-paech/livecodebench-code_generation_lite", split="test")    
    except Exception:
        # Fallback if Hugging Face mapped it to 'train'
        dataset = load_dataset("sam-paech/livecodebench-code_generation_lite", split="release_latest")

    # Time-Travel Eval: Take the absolute newest problems added to the benchmark
    # to guarantee Llama 3.1 has never seen them during pre-training.
    num_to_evaluate = min(args.num_samples, len(dataset))
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

    print(f"\n--- Generating {num_to_evaluate} LCB Predictions ---")
    results = []

    for item in tqdm(dataset):
        # Handle LCB schema variations gracefully
        task_id = item.get("id", item.get("question_id", "unknown_id"))
        problem_statement = item.get("question_content", item.get("problem", ""))
        starter_code = item.get("starter_code", "")

        # Map to our strict algorithmic prompt structure
        user_content = f"{problem_statement}\n\n"
        if starter_code:
            user_content += f"Please complete the following starter code:\n```python\n{starter_code}\n```"

        messages = [
            {
                "role": "system", 
                "content": "You are an expert competitive programmer. Read the problem statement and write a perfectly self-contained Python script to solve it. Output nothing but the pure Python code."
            },
            {
                "role": "user", 
                "content": user_content
            }
        ]
        
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=1024, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        clean_code = extract_code(generated_text)

        results.append({
            "task_id": task_id,
            "problem": problem_statement,
            "starter_code": starter_code,
            "generated_text": generated_text,
            "extracted_code": clean_code
        })

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nSuccess! Saved {len(results)} predictions to {args.output_file}")

if __name__ == "__main__":
    main()