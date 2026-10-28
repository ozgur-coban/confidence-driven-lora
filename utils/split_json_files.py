import json
import os
import math

def split_json_files(input_dir: str, output_dir: str, num_splits: int = 6):
    """
    Reads all JSON files in the input directory, splits their contents 
    into 'num_splits' equal parts, and saves them to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    for filename in json_files:
        input_path = os.path.join(input_dir, filename)
        
        with open(input_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping {filename}: Invalid JSON.")
                continue

        # Ensure the data is a list of objects
        if not isinstance(data, list):
            print(f"Skipping {filename}: Data is not a list of objects.")
            continue

        total_objects = len(data)
        if total_objects == 0:
            print(f"Skipping {filename}: File is empty.")
            continue

        print(f"Processing '{filename}' ({total_objects} objects)...")

        # Calculate base chunk size and how many chunks need an extra item (the remainder)
        base_size, remainder = divmod(total_objects, num_splits)
        
        start_idx = 0
        base_name, ext = os.path.splitext(filename)

        for i in range(num_splits):
            # Distribute the remainder across the first 'remainder' chunks
            current_chunk_size = base_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size
            
            chunk_data = data[start_idx:end_idx]
            
            # Only save if there is data in the chunk (handles cases where objects < num_splits)
            if chunk_data:
                output_filename = f"{base_name}_part{i+1}{ext}"
                output_path = os.path.join(output_dir, output_filename)
                
                with open(output_path, "w", encoding="utf-8") as out_f:
                    json.dump(chunk_data, out_f, indent=2, ensure_ascii=False)
                
            start_idx = end_idx

if __name__ == "__main__":
    # Update these paths to match your folder structure
    split_json_files(
        input_dir="./data/full_run_results/gemini_evaluation_format/routed_adapter",
        output_dir="./data/full_run_results/gemini_evaluation_format/routed_adapter/split_files",
        num_splits=6
    )