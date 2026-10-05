import json
import os
from collections import defaultdict
from typing import Any, Dict, List

def prepare_eval_files_from_dir(
    input_dir: str, 
    output_dir: str = "gemini_evaluation_format"
):
    """Reads all JSON files from an input directory, strips biasing metadata,
    assigns unique composite eval_ids, and saves grouped files named by {configuration}_{dataset}.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Locate all JSON files in the input directory
    json_files = sorted(
        [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(".json")
        ]
    )

    if not json_files:
        raise ValueError(f"No .json files found in directory: {input_dir}")

    print(f"Found {len(json_files)} JSON files in '{input_dir}'. Loading...")

    # 2. Ingest, normalize, and group records
    # Using a defaultdict to map "configuration_dataset" -> [list of clean records]
    grouped_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_ids = set()
    total_items = 0

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON in {file_path}. Skipping.")
                continue

        # Handle either a list of records or a single record dict
        items = data if isinstance(data, list) else [data]

        for item in items:
            # Extract composite keys
            dataset = item.get("dataset", "unknown")
            config = item.get("configuration", "default")
            prompt_id = item.get("prompt_id", "unknown_id")
            
            eval_id = f"{dataset}__{config}__{prompt_id}"

            # Guard against accidental duplicate rows across files
            if eval_id in seen_ids:
                print(f"Warning: Duplicate record detected and skipped: {eval_id}")
                continue
            seen_ids.add(eval_id)

            clean_record = {
                "eval_id": eval_id,
                "prompt": (item.get("prompt") or "").strip(),
                "ground_truth_answer": (
                    item.get("ground_truth_answer") or ""
                ).strip(),
                "generated_response": (
                    item.get("generated_response") or ""
                ).strip(),
            }
            
            # Determine the output filename key: configuration_dataset
            group_key = f"{config}_{dataset}"
            grouped_records[group_key].append(clean_record)
            total_items += 1

    print(f"Loaded {total_items} unique records across all files.")

    # 3. Save grouped records into separate files
    for group_key, records in grouped_records.items():
        output_filename = f"{group_key}.json"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(records, out_f, indent=2, ensure_ascii=False)

        print(f"Saved {len(records)} records to: {output_path}")

if __name__ == "__main__":
    prepare_eval_files_from_dir(
        input_dir="./data/full_run_results/raw", 
        output_dir="./data/full_run_results/gemini_evaluation_format"
    )