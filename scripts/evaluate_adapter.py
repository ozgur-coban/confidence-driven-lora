import torch
import argparse
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained LoRA adapter on GSM8K.")
    parser.add_argument("--base_model_id", type=str, default="NousResearch/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="./artifacts/adapters/adapter_reasoning/final")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of test samples to evaluate.")
    return parser.parse_args()

def extract_answer(text: str) -> str:
    """Extracts the final integer answer occurring after ####."""
    match = re.search(r'####\s*(-?[\d,]+)', text)
    return match.group(1).replace(',', '') if match else "Extraction Failed"

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. Loading Tokenizer and Data...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    dataset = load_dataset("openai/gsm8k", "main")
    test_data = dataset["test"].select(range(args.num_samples))

    print(f"2. Loading Base Model ({args.base_model_id})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print(f"3. Injecting LoRA Adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval() # Set to evaluation mode

    print("\n--- Starting Evaluation ---")
    
    for i, example in enumerate(test_data):
        print(f"\n[Problem {i+1}]")
        print(f"Question: {example['question']}")
        
        # 1. Format the prompt using the exact chat template from training
        messages = [
            {"role": "system", "content": "You are a mathematical reasoning assistant. Solve the problem step-by-step and provide the final answer after ####."},
            {"role": "user", "content": example["question"]}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True # Tells Llama 3 to append the assistant header token
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # 2. Generate with Greedy Decoding
        # Temperature 0.0 forces the model to pick the absolute highest probability token.
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 3. Decode and Parse
        # Slice the output to only include the newly generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        extracted_ans = extract_answer(response)
        ground_truth = extract_answer(example["answer"])

        print(f"\nModel Output:\n{response}")
        print("-" * 20)
        print(f"Extracted Model Answer: {extracted_ans}")
        print(f"Ground Truth Answer:    {ground_truth}")
        print(f"Correct:                {extracted_ans == ground_truth}")
        print("=" * 50)

if __name__ == "__main__":
    main()