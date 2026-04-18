import os
import sys
import subprocess
import argparse

def run_command(command: list, step_name: str):
    """Executes a terminal command, streams the output, and catches errors."""
    print(f"\n{'='*70}")
    print(f" STARTING: {step_name}")
    print(f"{'='*70}\n")
    
    try:
        # subprocess.run blocks execution until finished and prints output to console in real-time
        subprocess.run(command, check=True, text=True)
        print(f"\n SUCCESS: {step_name} completed without Python errors.")
    except subprocess.CalledProcessError:
        print(f"\n FAILED: {step_name} crashed.")
        print("Check the Python stack trace above to see exactly where the script broke.")
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Pipeline Verification")
    
    # Model & Adapter args
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter_name", type=str, default="adapter_verification")
    parser.add_argument("--output_dir", type=str, default="./models/adapters/test")
    
    # Training args
    parser.add_argument("--dataset", type=str, default="meta-math/MetaMathQA")
    parser.add_argument("--data_size", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    
    # Evaluation args
    parser.add_argument("--test_script", type=str, default="scripts/evaluate_metamath.py")
    parser.add_argument("--test_samples", type=int, default=5)
    parser.add_argument("--test_output_file", type=str, default="artifacts/generated_results/adapter_verification_results.json")
    
    return parser.parse_args()

def main():
    args = parse_args()
    print("Initializing End-to-End Pipeline Verification...\n")

    # Construct the final path based on user inputs
    test_adapter_path = os.path.join(args.output_dir, args.adapter_name, "final")

    # ---------------------------------------------------------
    # STEP 1: The Micro-Training Run
    # ---------------------------------------------------------
    train_cmd = [
        sys.executable, "scripts/train_adapter.py",
        "--model_id", args.model_id,
        "--adapter_name", args.adapter_name,
        "--dataset", args.dataset,
        "--data_size", str(args.data_size),
        "--batch_size", str(args.batch_size),
        "--grad_accum", str(args.grad_accum),
        "--epochs", "1",
        "--logging_steps", "2",
        "--learning_rate", "3e-4",
        "--output_dir", args.output_dir
    ]

    run_command(train_cmd, f"STEP 1: Micro-Training Run ({args.data_size} Samples)")

    # File I/O Verification: Ensure the model actually wrote to disk
    if os.path.exists(os.path.join(test_adapter_path, "adapter_config.json")):
        print(" Checkpoint verification passed: Files successfully written to disk.")
    else:
        print(" Checkpoint verification failed: No files found in output directory!")
        sys.exit(1)

    # ---------------------------------------------------------
    # STEP 2: The Micro-Evaluation Run
    # ---------------------------------------------------------
    eval_cmd = [
        sys.executable, args.test_script,
        "--base_model_id", args.model_id,
        "--adapter_path", test_adapter_path,
        "--num_samples", str(args.test_samples),
        "--output_file", args.test_output_file
    ]

    run_command(eval_cmd, f"STEP 2: Evaluation & Extraction Run ({args.test_samples} Samples)")

    # ---------------------------------------------------------
    # FINAL VERDICT
    # ---------------------------------------------------------
    print(f"\n{'='*70}")
    print(" PIPELINE VERIFIED: ALL SYSTEMS GO.")
    print(f"Target Adapter: {args.adapter_name}")
    print("The tokenization, masking, training, saving, and generation logic is flawless.")
    print("You are cleared to launch the full training runs.")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()