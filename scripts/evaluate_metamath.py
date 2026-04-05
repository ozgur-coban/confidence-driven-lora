import torch
import argparse
import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained LoRA adapter or base model on MetaMathQA.")
    parser.add_argument("--base_model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to the trained adapter. If omitted, runs base model.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples to evaluate. Set to -1 for the full test set.")
    parser.add_argument("--output_file", type=str, default="artifacts/generated_results/metamath_eval_results.json")
    return parser.parse_args()

def extract_model_answer(text: str) -> str:
    """Extracts the final answer occurring after ####."""
    if "####" in text:
        return text.split("####")[-1].strip()
    return "Extraction Failed"

def extract_ground_truth(text: str) -> str:
    """Extracts the explicit answer provided after 'The answer is: ' in MetaMathQA."""
    if "The answer is:" in text:
        return text.split("The answer is:")[-1].strip()
    return "Extraction Failed"

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. Loading Tokenizer and Rebuilding Dataset Splits...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    
    # Replicate the exact preprocessor filtering and splitting to prevent data leakage
    filtered_dataset = dataset.filter(
        lambda x: (len(str(x["query"])) + len(str(x["response"]))) < 1800
    )
    dataset_15k = filtered_dataset.shuffle(seed=42).select(range(15000))
    train_test_split = dataset_15k.train_test_split(test_size=5000, seed=42)
    val_test_split = train_test_split['test'].train_test_split(test_size=3000, seed=42)
    
    # Isolate the test split and handle -1 for the full available test set
    full_test_set = val_test_split['test']
    num_to_evaluate = len(full_test_set) if args.num_samples == -1 else min(args.num_samples, len(full_test_set))
    test_data = full_test_set.shuffle(seed=42).select(range(num_to_evaluate))

    print(f"2. Loading Base Model: {args.base_model_id} in 4-bit...")
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
        messages = [
            {"role": "system", "content": "You are a mathematical reasoning assistant. Solve the problem step-by-step and provide the final answer after ####."},
            {"role": "user", "content": example["query"]}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512, # Adjusted to 512 based on the filtered dataset length
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
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

        results.append({
            "id": i,
            "query": example["query"],
            "ground_truth": ground_truth,
            "predicted_answer": extracted_ans,
            "raw_output": response,
            "is_correct": is_correct
        })

    # Print Final Statistics
    accuracy = (correct_count / len(test_data)) * 100
    print("\n" + "="*50)
    print("FINAL EVALUATION METRICS")
    print("="*50)
    print(f"Total Samples Tested:  {len(test_data)}")
    print(f"Correct Answers:       {correct_count}")
    print(f"Failed Extractions:    {failed_extraction_count}")
    print(f"Overall Accuracy:      {accuracy:.2f}%")
    print("="*50)

    # Save detailed logs
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nDetailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()