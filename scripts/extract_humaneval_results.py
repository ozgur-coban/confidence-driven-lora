import json
import subprocess
import tempfile
import os
from tqdm import tqdm

def main():
    prediction_file = "humaneval_predictions.json" 
    
    if not os.path.exists(prediction_file):
        print(f"Error: Please upload '{prediction_file}' to the Colab environment.")
        return

    with open(prediction_file, "r") as f:
        data = json.load(f)

    print(f"--- Starting Air-Gapped Execution on {len(data)} Samples ---")
    
    passed = 0
    failed = 0

    for item in tqdm(data, desc="Evaluating Code"):
        # 1. Stitch the executable script together
        # We combine the model's code, the assertion tests, and the trigger function
        full_script = f"{item['extracted_code']}\n{item['test']}\ncheck({item['entry_point']})\n"

        # 2. Write to a temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False) as tmp:
            tmp.write(full_script)
            tmp_path = tmp.name

        # 3. Execute in an isolated subprocess
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
            else:
                failed += 1
                # Print exactly why it failed for debugging
                error_msg = result.stderr.strip().split('\n')[-1] if result.stderr else "Unknown Error"
                tqdm.write(f"\n[Task {item['task_id']}] FAILED: {error_msg}")
                
        except subprocess.TimeoutExpired:
            failed += 1
            tqdm.write(f"\n[Task {item['task_id']}] FAILED: Timeout (Infinite Loop)")
            
        finally:
            # Clean up the temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # 4. Final Scoring
    accuracy = (passed / len(data)) * 100

    print("\n" + "="*50)
    print("FINAL EXECUTION METRICS (pass@1)")
    print("="*50)
    print(f"Total Samples Tested:   {len(data)}")
    print(f"Correct Executions:     {passed}")
    print(f"Failed Executions:      {failed}")
    print(f"Functional Accuracy:    {accuracy:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()