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
    parser = argparse.ArgumentParser(description="Generate HumanEval predictions using a trained LoRA adapter.")
    parser.add_argument("--base_model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained Code LoRA adapter.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate (max 164).")
    parser.add_argument("--output_file", type=str, default="artifacts/generated_results/humaneval_predictions.json")
    return parser.parse_args()

def extract_code(text: str) -> str:
    """
    Attempts to extract code from markdown backticks.
    If no backticks are found, assumes the entire text is raw code.
    """
    match = re.search(r'```(?:python)?(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. Loading HumanEval Dataset...")
    dataset = load_dataset("openai_humaneval", split="test")
    
    # Restrict to the requested number of samples to save time
    num_to_evaluate = min(args.num_samples, len(dataset))
    dataset = dataset.select(range(num_to_evaluate))

    print(f"2. Loading Base Model and Injecting Adapter: {args.adapter_path}")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, 
        quantization_config=quant_config, 
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)

    print(f"\n--- Generating {num_to_evaluate} Predictions ---")
    results = []

    for item in tqdm(dataset):
        task_id = item["task_id"]
        prompt_text = item["prompt"]
        
        user_content = (
            "Complete the following Python function. "
            "Output the entire complete function, including the original signature and docstring. "
            "Do not provide explanations, only the code.\n\n"
            f"```python\n{prompt_text}\n```"
        )

        messages = [
            {"role": "system", "content": "You are an expert programming assistant. Write clean, efficient, and well-documented Python code."},
            {"role": "user", "content": user_content}
        ]
        
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        clean_code = extract_code(generated_text)

        results.append({
            "task_id": task_id,
            "prompt": prompt_text,
            "canonical_solution": item["canonical_solution"],
            "test": item["test"],
            "entry_point": item["entry_point"],
            "generated_text": generated_text,
            "extracted_code": clean_code
        })

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nSuccess! Saved {len(results)} predictions to {args.output_file}")
    print("Move this file to an isolated Colab environment for safe execution.")

if __name__ == "__main__":
    main()