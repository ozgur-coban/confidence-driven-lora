import torch
import argparse
import json
import os
import re
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from datasets import load_dataset
from unsloth import FastLanguageModel

# ---------------------------------------------------------
# 1. ADDED: Custom Stopping Criteria
# ---------------------------------------------------------
class KeywordStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words: list, tokenizer):
        self.stop_words = stop_words
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Decode the last 15 tokens to check for tripwires efficiently.
        generated_text = self.tokenizer.decode(input_ids[0][-15:], skip_special_tokens=True)
        
        for word in self.stop_words:
            if word in generated_text:
                return True # Trigger the stop
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained LoRA adapter or base model on MetaMathQA.")
    parser.add_argument("--base_model_id", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to the trained adapter. If omitted, runs base model.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples to evaluate. Set to -1 for the full test set.")
    parser.add_argument("--output_file", type=str, default="artifacts/generated_results/metamath_eval_results.json")
    return parser.parse_args()

def clean_math_string(s: str) -> str:
    """Normalizes math outputs: safely unwraps \\boxed{}, strips $, cleans commas, and parses sentences."""
    s = s.strip()
    
    # Remove wrapping #### or spaces
    s = re.sub(r'^#+', '', s)
    s = re.sub(r'#+$', '', s)
    s = s.strip()
    
    # Safely extract from \boxed{...} handling nested brackets
    if '\\boxed{' in s:
        start = s.find('\\boxed{') + 7
        depth = 1
        for i in range(start, len(s)):
            if s[i] == '{': depth += 1
            elif s[i] == '}': depth -= 1
            if depth == 0:
                s = s[start:i]
                break
                
    s = s.strip('$ \n\t.')
    
    # Remove conversational prefixes
    s = re.sub(r'^(The )?(final )?answer is\s*:?\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^Therefore,?\s*(the )?.*?\s+is\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^So,?\s*(the )?.*?\s+is\s*', '', s, flags=re.IGNORECASE)
    
    s = s.strip('$ \n\t.')
    
    # If it is a sentence containing the answer (e.g. "The theater sold 93 tickets")
    if len(s) > 10 and not any(op in s for op in ['+', '-', '*', '/', '\\', '^', '=', 'x', 'y', 'z']):
        nums = re.findall(r'-?[\d,]+(?:\.\d+)?', s)
        if nums:
            s = nums[-1]
            
    # Remove commas in numbers for comparison (e.g., 6,318 -> 6318)
    if re.match(r'^-?[\d,]+(?:\.\d+)?$', s):
        s = s.replace(',', '')
        
    return s.strip()

def extract_model_answer(text: str) -> str:
    """Extracts the final answer, handling both strict LoRA formatting and base model fallbacks."""
    ans = "Extraction Failed"
    
    if "####" in text:
        parts = text.split("####")
        ans = parts[1] if len(parts) > 1 and parts[1].strip() else parts[-1]
    elif "The final answer is:" in text:
        ans = text.split("The final answer is:")[-1]
    elif "The answer is:" in text:
        ans = text.split("The answer is:")[-1]
    else:
        numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
        if numbers:
            ans = numbers[-1]
            
    if ans == "Extraction Failed":
        return ans
        
    return clean_math_string(ans)

def extract_ground_truth(text: str) -> str:
    """Extracts the explicit answer provided after 'The answer is: ' in MetaMathQA."""
    if "The answer is:" in text:
        ans = text.split("The answer is:")[-1].strip()
        return clean_math_string(ans)
    return "Extraction Failed"

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. Loading and Rebuilding Dataset Splits...")
    # FIX: Removed the redundant AutoTokenizer initialization
    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    
    filtered_dataset = dataset.filter(
        lambda x: (len(str(x["query"])) + len(str(x["response"]))) < 1800
    )
    dataset_15k = filtered_dataset.shuffle(seed=42).select(range(15000))
    train_test_split = dataset_15k.train_test_split(test_size=5000, seed=42)
    val_test_split = train_test_split['test'].train_test_split(test_size=3000, seed=42)
    
    full_test_set = val_test_split['test']
    num_to_evaluate = len(full_test_set) if args.num_samples == -1 else min(args.num_samples, len(full_test_set))
    test_data = full_test_set.shuffle(seed=42).select(range(num_to_evaluate))

    print(f"2. Loading Base Model: {args.base_model_id} via Unsloth in 4-bit...")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.base_model_id,
        max_seq_length = 512,
        dtype = None, 
        load_in_4bit = True,
    )
    # FIX: Removed the forceful get_chat_template injection

    if args.adapter_path:
        print(f"3. Injecting LoRA Adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    else:
        print("3. No adapter path provided. Bypassing adapter: Evaluating Base Model only...")
        model = base_model
        
    FastLanguageModel.for_inference(model) 
    model.eval()

    print(f"\n--- Starting Evaluation on {len(test_data)} Samples ---")
    
    results = []
    correct_count = 0
    failed_extraction_count = 0

    # ---------------------------------------------------------
    # 2. ADDED: Initialize the Tripwires
    # ---------------------------------------------------------
    stop_words = ["\nQuestion:", "\nHuman:", "\nRequirement:", "\nCase/Query:"]
    stop_criteria = KeywordStoppingCriteria(stop_words, tokenizer)
    stopping_criteria_list = StoppingCriteriaList([stop_criteria])

    for i, example in enumerate(tqdm(test_data, desc="Evaluating")):
        
        # ---------------------------------------------------------
        # 3. FIX: Prompt formatting matches the training template exactly
        # ---------------------------------------------------------
        prompt = f"{example['query']}\n\nAnswer: "
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria_list # Included criteria here
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

    accuracy = (correct_count / len(test_data)) * 100
    print("\n" + "="*50)
    print("FINAL EVALUATION METRICS")
    print("="*50)
    print(f"Total Samples Tested:  {len(test_data)}")
    print(f"Correct Answers:       {correct_count}")
    print(f"Failed Extractions:    {failed_extraction_count}")
    print(f"Overall Accuracy:      {accuracy:.2f}%")
    print("="*50)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nDetailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()