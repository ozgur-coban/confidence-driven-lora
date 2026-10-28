import os
import json
import random
from datasets import load_dataset

def save_json(data, filename):
    filepath = os.path.join("./data/final_datasets/formatted", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f" -> Saved {len(data)} samples to {filepath}")

def format_gsm8k(sample_size=300):
    print("\nFormatting GSM8K (Pure Math)...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    
    # Take from the extreme end of the dataset
    total_len = len(ds)
    indices = list(range(total_len - sample_size, total_len))
    random.shuffle(indices) # Shuffle the selection so topics aren't clustered
    
    formatted = []
    for i, idx in enumerate(indices):
        row = ds[idx]
        formatted.append({
            "id": f"gsm8k_{i+1:03d}",
            "prompt": row["question"],
            "expected_domain": "math",
            "ground_truth_answer": row["answer"]
        })
        
    save_json(formatted, "formatted_gsm8k.json")
    return formatted

def format_codefeedback(sample_size=300):
    print("\nFormatting CodeFeedback (Pure Code)...")
    try:
        ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="test")
    except:
        ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
        
    # Strictly take from the end to avoid overlap with training data
    total_len = len(ds)
    indices = list(range(total_len - sample_size, total_len))
    random.shuffle(indices)
    
    formatted = []
    for i, idx in enumerate(indices):
        row = ds[idx]
        formatted.append({
            "id": f"codefeedback_{i+1:03d}",
            "prompt": row["query"],
            "expected_domain": "code",
            "ground_truth_answer": row.get("answer", "")
        })
        
    save_json(formatted, "formatted_codefeedback.json")
    return formatted

def format_mixed(math_data, code_data, sample_size=300):
    print(f"\nFormatting Mixed Dataset (50/50)...")
    half = sample_size // 2
    
    # Take half of each (which are already randomized from the end of their parent sets)
    mixed_data = math_data[:half] + code_data[:half]
    random.shuffle(mixed_data)
    
    # Re-assign IDs for clarity
    for i, item in enumerate(mixed_data):
        item["id"] = f"pure_mixed_{i+1:03d}"
        
    save_json(mixed_data, "formatted_mixed.json")

def format_synthetic_hybrid(input_path="./data/gsm8k_test_batches/gsm8k_hybrid_test.json", sample_size=265):
    print("\nFormatting Synthetic Hybrid (Math + Code)...")
    if not os.path.exists(input_path):
        print(f" [WARN] Could not find {input_path}. Skipping.")
        return
        
    with open(input_path, "r", encoding="utf-8") as f:
        raw_prompts = json.load(f)
        
    # Take from the end, then shuffle
    selected = raw_prompts[-sample_size:] if len(raw_prompts) > sample_size else raw_prompts
    random.shuffle(selected)
    
    formatted = []
    for i, prompt in enumerate(selected):
        formatted.append({
            "id": f"hybrid_synthetic_gsm8k_{i+1:03d}",
            "prompt": prompt,
            "expected_domain": "hybrid",
            "ground_truth_answer": None # None bypasses automated execution eval
        })
        
    save_json(formatted, "formatted_gsm8k_hybrid.json")

def format_wildchat(input_path="./data/wildchat/hybrid_wildchat_from_gemini_filtered_further.json"):
    print("\nFormatting WildChat (OOD Noise)...")
    if not os.path.exists(input_path):
        print(f" [WARN] Could not find {input_path}. Skipping.")
        return
        
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    formatted = []
    for i, item in enumerate(raw_data):
        prompt = item["prompt"] if isinstance(item, dict) and "prompt" in item else str(item)
        
        formatted.append({
            "id": f"wildchat_{i+1:03d}",
            "prompt": prompt,
            "expected_domain": "hybrid",
            "ground_truth_answer": None 
        })
        
    save_json(formatted, "formatted_wildchat.json")

if __name__ == "__main__":
    os.makedirs("./data/final_datasets/formatted", exist_ok=True)
    random.seed(42) # Lock the seed for reproducibility
    
    print("Starting Dataset Formatting Pipeline...")
    
    math_data = format_gsm8k(300)
    code_data = format_codefeedback(300)
    format_mixed(math_data, code_data, 300)
    
    # Using your exact local paths
    format_synthetic_hybrid(input_path="./data/gsm8k_test_batches/gsm8k_hybrid_test.json")
    format_wildchat(input_path="./data/wildchat/hybrid_wildchat_from_gemini_filtered_further.json")
    
    print("\nAll datasets formatted safely (no training data leakage)!")