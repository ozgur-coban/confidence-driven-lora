import torch
import argparse
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained LoRA adapter on GSM8K.")
    parser.add_argument("--base_model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to the trained adapter. If omitted, runs base model.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples to evaluate.")
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
    
    # Shuffle with a fixed seed to get a consistent random subset of 100 problems
    test_data = dataset["test"].shuffle(seed=42).select(range(args.num_samples))

    print(f"2. Loading Base Model ({args.base_model_id}) in 4-bit...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        quantization_config=quant_config,
        device_map="auto"
    )

# comment out to test with LoRA
    if args.adapter_path:
        print(f"3. Injecting LoRA Adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    else:
        print("3. No adapter path provided. Bypassing adapter: Evaluating Base Model only...")
        model = base_model
    
    model.eval()
    

    

    print(f"\n--- Starting Evaluation on {args.num_samples} Samples ---")
    
    correct_count = 0
    failed_extraction_count = 0

    for i, example in enumerate(tqdm(test_data, desc="Evaluating")):
        messages = [
            {"role": "system", "content": "You are a mathematical reasoning assistant. Solve the problem step-by-step and provide the final answer after ####."},
            {"role": "user", "content": example["question"]}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        extracted_ans = extract_answer(response)
        ground_truth = extract_answer(example["answer"])

        is_correct = extracted_ans == ground_truth
        if extracted_ans == "Extraction Failed":
            failed_extraction_count += 1
        elif is_correct:
            correct_count += 1

        # Print the outputs directly to the console during the loop
        tqdm.write(f"\n[Problem {i+1}]")
        tqdm.write(f"Question: {example['question']}")
        tqdm.write(f"\nModel Output:\n{response}")
        tqdm.write("-" * 20)
        tqdm.write(f"Extracted Model Answer: {extracted_ans}")
        tqdm.write(f"Ground Truth Answer:    {ground_truth}")
        tqdm.write(f"Correct:                {is_correct}")
        tqdm.write("=" * 50)

    # Print Final Statistics
    accuracy = (correct_count / args.num_samples) * 100
    print("\n" + "="*50)
    print("FINAL EVALUATION METRICS")
    print("="*50)
    print(f"Total Samples Tested:  {args.num_samples}")
    print(f"Correct Answers:       {correct_count}")
    print(f"Failed Extractions:    {failed_extraction_count}")
    print(f"Overall Accuracy:      {accuracy:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()