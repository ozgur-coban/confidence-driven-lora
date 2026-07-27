import json
import os
import glob
import random

def stitch_and_split_synthetic_batches(
    input_dir="./gsm8k_synthetic_batches",
    train_output_path="./data/gsm8k_hybrid_train.json",
    test_output_path="./data/gsm8k_hybrid_test.json",
    split_ratio=0.8,
    seed=42
):
    """
    Stitches all synthetic JSON batch files in input_dir, cleans up formatting,
    performs a reproducible train/test split, and exports the final JSON files.
    """
    random.seed(seed)
    
    # Match any JSON batch file in the input directory
    file_paths = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    
    if not file_paths:
        raise FileNotFoundError(f"No JSON files found in input directory: '{input_dir}'")

    print(f"Found {len(file_paths)} batch files to aggregate.")

    all_prompts = []
    failed_files = []

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                
            # Strip markdown code blocks if present in the raw file
            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
                
            batch_data = json.loads(content)
            
            if isinstance(batch_data, list):
                valid_prompts = [str(p).strip() for p in batch_data if isinstance(p, str) and p.strip()]
                all_prompts.extend(valid_prompts)
            else:
                print(f"Warning: File {file_path} did not contain a top-level JSON list. Skipping.")
                failed_files.append(file_path)

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            failed_files.append(file_path)

    print(f"Extracted {len(all_prompts)} raw prompts.")

    # Deduplicate prompts while maintaining initial order
    unique_prompts = []
    seen = set()
    for prompt in all_prompts:
        if prompt not in seen:
            seen.add(prompt)
            unique_prompts.append(prompt)

    if len(unique_prompts) < len(all_prompts):
        print(f"Removed {len(all_prompts) - len(unique_prompts)} duplicate prompts.")

    # Shuffle dataset deterministically
    random.shuffle(unique_prompts)

    # Perform 80/20 train/test split
    split_index = int(len(unique_prompts) * split_ratio)
    train_prompts = unique_prompts[:split_index]
    test_prompts = unique_prompts[split_index:]

    # Ensure target output directories exist
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_output_path), exist_ok=True)

    # Save train split
    with open(train_output_path, "w", encoding="utf-8") as f:
        json.dump(train_prompts, f, indent=4)

    # Save test split
    with open(test_output_path, "w", encoding="utf-8") as f:
        json.dump(test_prompts, f, indent=4)

    print("\n--- Dataset Split Summary ---")
    print(f"Total Unique Synthetic Prompts : {len(unique_prompts)}")
    print(f"Train Set ({int(split_ratio*100)}%)               : {len(train_prompts)} -> '{train_output_path}'")
    print(f"Test Set ({int((1-split_ratio)*100)}%)                : {len(test_prompts)} -> '{test_output_path}'")
    
    if failed_files:
        print(f"\nSkipped files due to formatting issues ({len(failed_files)}): {failed_files}")

if __name__ == "__main__":
    stitch_and_split_synthetic_batches(
        input_dir="./data/gsm8k_test_batches/gsm8k_hybrid_batches",
        train_output_path="./data/gsm8k_test_batches/gsm8k_hybrid_train.json",
        test_output_path="./data/gsm8k_test_batches/gsm8k_hybrid_test.json"
    )