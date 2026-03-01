import json
import argparse
import os
import subprocess
import tempfile
import zlib
import pickle
import base64
from tqdm import tqdm
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LiveCodeBench predictions using isolated execution.")
    parser.add_argument("--predictions_file", type=str, required=True, help="Path to the generated predictions JSON.")
    parser.add_argument("--output_file", type=str, default="artifacts/generated_results/lcb_eval_results.json")
    parser.add_argument("--timeout", type=int, default=5, help="Execution timeout in seconds per test case.")
    return parser.parse_args()

def load_lcb_dataset():
    """Loads the original dataset to fetch the hidden test cases."""
    try:
        dataset = load_dataset("livecodebench/code_generation_lite", split="test")
    except Exception:
        dataset = load_dataset("livecodebench/code_generation_lite", split="train")
    
    # Create a quick lookup dictionary by task_id
    problem_dict = {}
    for item in dataset:
        task_id = item.get("id", item.get("question_id"))
        problem_dict[task_id] = item
    return problem_dict

def decode_private_tests(private_test_str):
    """Decodes LCB's compressed private test cases."""
    if not private_test_str:
        return []
    try:
        return json.loads(private_test_str)
    except:
        try:
            return json.loads(pickle.loads(zlib.decompress(base64.b64decode(private_test_str.encode("utf-8")))))
        except Exception as e:
            print(f"Failed to decode private tests: {e}")
            return []

def run_functional_test(code, test_input, expected_output, func_name, timeout):
    """Executes LeetCode-style functional code."""
    # We write a dynamic wrapper to instantiate the class and call the method
    wrapper = f"""
import sys, json, math, collections, itertools
from typing import *

{code}

if __name__ == "__main__":
    try:
        # LCB functional inputs are often string-represented JSON arrays
        raw_input = sys.stdin.read().strip()
        args = [json.loads(arg) for arg in json.loads(raw_input)]
        
        func_name = "{func_name}"
        if "Solution" in globals():
            obj = Solution()
            func = getattr(obj, func_name)
        else:
            func = globals()[func_name]
            
        res = func(*args)
        print(json.dumps(res))
    except Exception as e:
        print(f"EXECUTION_ERROR: {{e}}")
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(wrapper)
        temp_file_path = f.name

    try:
        # Pass the arguments via standard input as a JSON list
        result = subprocess.run(
            ["python3", temp_file_path],
            input=test_input,
            text=True,
            capture_output=True,
            timeout=timeout
        )
        output = result.stdout.strip()
        
        # Check against expected output (which is also JSON formatted in LCB)
        try:
            return json.loads(output) == json.loads(expected_output)
        except:
            return output == expected_output.strip()

    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        os.remove(temp_file_path)

def run_stdin_test(code, test_input, expected_output, timeout):
    """Executes CodeForces-style standard I/O code."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file_path = f.name

    try:
        result = subprocess.run(
            ["python3", temp_file_path],
            input=test_input,
            text=True,
            capture_output=True,
            timeout=timeout
        )
        return result.stdout.strip() == expected_output.strip()
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        os.remove(temp_file_path)

def main():
    args = parse_args()
    
    print("1. Loading evaluation data and decoding hidden test cases...")
    dataset_dict = load_lcb_dataset()
    
    with open(args.predictions_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    print(f"\n--- Starting Air-Gapped Execution on {len(predictions)} Samples ---")
    
    results = []
    correct_count = 0

    for idx, pred in enumerate(tqdm(predictions, desc="Evaluating Code")):
        task_id = pred["task_id"]
        code = pred["extracted_code"]
        
        if task_id not in dataset_dict:
            continue
            
        problem_data = dataset_dict[task_id]
        metadata = json.loads(problem_data.get("metadata", "{}"))
        test_type = metadata.get("testtype", "stdin") # Fallback to stdin
        func_name = metadata.get("func_name", "")
        
        # Aggregate public and private test cases
        public_tests = json.loads(problem_data.get("public_test_cases", "[]"))
        private_tests = decode_private_tests(problem_data.get("private_test_cases", ""))
        all_tests = public_tests + private_tests
        
        if not all_tests:
            continue

        passed_all = True
        for test in all_tests:
            test_in = test.get("input", "")
            test_out = test.get("output", "")
            
            if test_type == "functional":
                passed = run_functional_test(code, test_in, test_out, func_name, args.timeout)
            else:
                passed = run_stdin_test(code, test_in, test_out, args.timeout)
                
            if not passed:
                passed_all = False
                break # Stop at the first failed test case to save time

        if passed_all:
            correct_count += 1
            
        results.append({
            "task_id": task_id,
            "test_type": test_type,
            "passed": passed_all
        })

    accuracy = (correct_count / len(results)) * 100 if results else 0.0

    print("\n" + "="*50)
    print("FINAL LIVECODEBENCH EXECUTION METRICS (pass@1)")
    print("="*50)
    print(f"Total Samples Tested:   {len(results)}")
    print(f"Correct Executions:     {correct_count}")
    print(f"Failed Executions:      {len(results) - correct_count}")
    print(f"Functional Accuracy:    {accuracy:.2f}%")
    print("="*50)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()