import os
import json
import re

def extract_labels_via_regex(filepath: str) -> dict:
    """Fallback text scraper for broken JSON files with unescaped quotes."""
    extracted = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Split the document by "eval_id" to evaluate one block at a time
    blocks = content.split('"eval_id"')
    
    for block in blocks[1:]: # Skip the chunk before the first eval_id
        # 1. Grab the ID string right after "eval_id"
        id_match = re.search(r'^\s*:\s*"([^"]+)"', block)
        if not id_match:
            continue
        eval_id = id_match.group(1)
        
        # 2. Grab the true/false value for is_correct in this same block
        bool_match = re.search(r'"is_correct"\s*:\s*(true|false)', block, re.IGNORECASE)
        if bool_match:
            is_correct_str = bool_match.group(1).lower()
            extracted[eval_id] = (is_correct_str == 'true')
            
    return extracted

def merge_gemini_evaluations(raw_dir: str, gemini_dir: str, output_dir: str):
    gemini_mapping = {}
    
    print("Scanning Gemini evaluation files...")
    for root, _, files in os.walk(gemini_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    # 1. Try standard JSON first
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        if "eval_id" in item and "is_correct" in item:
                            gemini_mapping[item["eval_id"]] = item["is_correct"]
                            
                except json.JSONDecodeError:
                    # 2. If JSON is broken, rescue the data using Regex
                    print(f"  [WARN] Broken JSON in {file} -> Rescuing via Regex...")
                    rescued_data = extract_labels_via_regex(file_path)
                    gemini_mapping.update(rescued_data)

    print(f"Loaded {len(gemini_mapping)} evaluated records from Gemini.\n")

    os.makedirs(output_dir, exist_ok=True)
    raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]
    
    if not raw_files:
        print(f"[!] No JSON files found in {raw_dir}")
        return

    print("Merging evaluations into raw files...")
    for file in raw_files:
        file_path = os.path.join(raw_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        items = raw_data if isinstance(raw_data, list) else [raw_data]
        updated_count = 0
        
        for item in items:
            dataset = item.get("dataset", "unknown")
            config = item.get("configuration", "default")
            prompt_id = item.get("prompt_id", "unknown_id")
            
            eval_id = f"{dataset}__{config}__{prompt_id}"
            
            if eval_id in gemini_mapping:
                item["execution_correct"] = gemini_mapping[eval_id]
                updated_count += 1
        
        output_path = os.path.join(output_dir, file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=4, ensure_ascii=False)
            
        print(f"  -> Saved {file} (Updated {updated_count}/{len(items)} records)")
        
    print(f"\nAll files successfully merged into: '{output_dir}'")

if __name__ == "__main__":
    merge_gemini_evaluations(
        raw_dir="./data/full_run_results/raw",
        gemini_dir="./data/full_run_results/gemini_results",
        output_dir="./data/full_run_results/final_merged_results"
    )