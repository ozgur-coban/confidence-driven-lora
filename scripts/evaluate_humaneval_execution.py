import json
import subprocess
import tempfile
import os
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HumanEval predictions.")
    parser.add_argument("--predictions_file", type=str, required=True, help="Path to predictions JSON.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save results JSON.")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.predictions_file):
        print(f"Error: Could not find '{args.predictions_file}'.")
        return

    with open(args.predictions_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"--- Starting Air-Gapped Execution on {len(data)} Samples ---")
    
    passed = 0
    failed = 0
    results_log = []

    for item in tqdm(data, desc="Evaluating Code"):
        # SAFETY FIX 1: Add standard library imports
        safety_imports = "import math\nimport collections\nfrom typing import *\n\n"
        
        # SAFETY FIX 2: Prepend prompt if Base Model only generated the body
        code_to_run = item.get('extracted_code', '')
        if "def " not in code_to_run:
             code_to_run = item.get('prompt', '') + "\n" + code_to_run

        # Stitch the executable script together
        full_script = f"{safety_imports}{code_to_run}\n{item['test']}\ncheck({item['entry_point']})\n"

        # Write to a temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False) as tmp:
            tmp.write(full_script)
            tmp_path = tmp.name

        passed_this_test = False
        error_msg = ""

        # Execute in an isolated subprocess
        try:
            result = subprocess.run(
                ["python", tmp_path], 
                capture_output=True, 
                text=True, 
                timeout=5 # Kills infinite while-loops
            )
            
            # Exit code 0 means no SyntaxErrors and all assertions passed
            if result.returncode == 0:
                passed += 1
                passed_this_test = True
            else:
                failed += 1
                error_msg = result.stderr.strip().split('\n')[-1] if result.stderr else "Unknown Error"
                tqdm.write(f"\n[Task {item['task_id']}] FAILED: {error_msg}")
                
        except subprocess.TimeoutExpired:
            failed += 1
            error_msg = "Timeout (Infinite Loop)"
            tqdm.write(f"\n[Task {item['task_id']}] FAILED: {error_msg}")
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Log for the output file
        results_log.append({
            "task_id": item["task_id"],
            "passed": passed_this_test,
            "error": error_msg if not passed_this_test else None
        })

    # Final Scoring
    accuracy = (passed / len(data)) * 100 if data else 0.0

    print("\n" + "="*50)
    print("FINAL HUMANEVAL EXECUTION METRICS (pass@1)")
    print("="*50)
    print(f"Total Samples Tested:   {len(data)}")
    print(f"Correct Executions:     {passed}")
    print(f"Failed Executions:      {failed}")
    print(f"Functional Accuracy:    {accuracy:.2f}%")
    print("="*50)

    # Save Results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=4)

if __name__ == "__main__":
    main()