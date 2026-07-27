from datasets import load_dataset
import os

def prepare_gsm8k_for_gemini(output_dir="./data/gsm8k_test_batches", batch_size=100):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load official GSM8K test set (~1,319 problems)
    print("Loading GSM8K test set...")
    ds = load_dataset("gsm8k", "main", split="test")
    
    current_batch = []
    batch_num = 1
    
    for idx, row in enumerate(ds):
        # Extract the question body
        question = row["question"].replace("\n", " ").strip()
        current_batch.append(f"[{idx}] {question}")
        
        # Save batch when reaching batch_size
        if len(current_batch) == batch_size or idx == len(ds) - 1:
            file_path = os.path.join(output_dir, f"batch_{batch_num}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(current_batch))
            
            print(f"Saved {len(current_batch)} problems to {file_path}")
            current_batch = []
            batch_num += 1

if __name__ == "__main__":
    prepare_gsm8k_for_gemini()