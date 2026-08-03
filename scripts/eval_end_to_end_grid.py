import os
import sys
import json
import time
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login

from src.router.meta_router import MetaRouter
from src.model_ops.inference_manager import InferenceManager
from src.eval.metrics import evaluate_math

def load_standardized_json(filepath, max_samples=None):
    if not os.path.exists(filepath):
        print(f"  [WARN] Missing dataset file: {filepath}")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_samples and max_samples > 0:
        return data[:max_samples]
    return data

def parse_args():
    parser = argparse.ArgumentParser(description="Parameterized End-to-End Evaluation Grid")
    
    # Dataset Selection
    parser.add_argument(
        "--datasets", nargs="+", 
        choices=["gsm8k", "codefeedback", "mixed", "hybrid", "wildchat", "all"],
        default=["all"], help="Which datasets to evaluate."
    )
    
    # Configuration Selection
    parser.add_argument(
        "--configs", nargs="+", 
        choices=["base", "pure_math", "pure_code", "hybrid_adapter", "routed", "all"],
        default=["all"], help="Which configurations (adapters/routers) to test."
    )
    
    # Adapter Paths
    parser.add_argument("--code_adapter", default="./models/adapters/adapter_code/code_codefeedback_3b_base_07.06.2026")
    parser.add_argument("--math_adapter", default="./models/adapters/adapter_math/math_gsm8k_main_3b_base_07.06.2026")
    parser.add_argument("--hybrid_adapter", default="./models/adapters/adapter_hybrid/math_code_ties_07.06.2026/hybrid")
    
    # Execution Modifiers
    parser.add_argument("--num_samples", type=int, default=-1, help="Limit samples per dataset for quick testing.")
    # parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate.")
    parser.add_argument("--out_dir", type=str, default="logs/eval_grid", help="Local directory for output JSONs.")
    
    # --- GOOGLE DRIVE INTEGRATION ---
    parser.add_argument("--drive_dir", type=str, default="/content/drive/MyDrive/capstone_eval_logs", 
                        help="Google Drive path to sync logs to (Only used if running in Colab).")
    
    return parser.parse_args()

def main():
    args = parse_args()
    load_dotenv()
    
    # --- DETECT COLAB & MOUNT DRIVE ---
    IN_COLAB = 'google.colab' in sys.modules
    if IN_COLAB:
        print("\n[!] Google Colab Environment Detected. Mounting Google Drive...")
        from google.colab import drive
        drive.mount('/content/drive')
        os.makedirs(args.drive_dir, exist_ok=True)
        print(f" -> Drive mounted. Logs will be synced to: {args.drive_dir}\n")

    if os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))

    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve "all" arguments
    all_datasets = ["gsm8k", "codefeedback", "mixed", "hybrid", "wildchat"]
    all_configs = ["base", "pure_math", "pure_code", "hybrid_adapter", "routed"]
    
    target_datasets = all_datasets if "all" in args.datasets else args.datasets
    target_configs = all_configs if "all" in args.configs else args.configs

    # 1. Boot Subsystems
    print("\n[1] Booting Subsystems...")
    router = None
    if "routed" in target_configs:
        router = MetaRouter()
        router.load()

    adapters = {
        "math": args.math_adapter,
        "code": args.code_adapter,
        "hybrid": args.hybrid_adapter
    }
    
    print("Loading LLM and LoRAs (This takes a moment)...")
    llm = InferenceManager(base_model_id="meta-llama/Llama-3.2-3B", adapter_paths=adapters)

    # 2. File Map (Adjust these to match your actual standardized file names)
    dataset_files = {
        "gsm8k": "data/final_datasets/formatted/formatted_gsm8k.json",
        "codefeedback": "data/final_datasets/formatted/formatted_codefeedback.json",
        "mixed": "data/final_datasets/formatted/formatted_mixed.json",
        "hybrid": "data/final_datasets/formatted/formatted_gsm8k_hybrid_restored.json",
        "wildchat": "data/final_datasets/formatted/formatted_wildchat_restored.json"
    }

    print("\n" + "="*50)
    print(f"STARTING EVALUATION GRID")
    print(f"Datasets: {target_datasets}")
    print(f"Configs:  {target_configs}")
    print("="*50)

    # 3. Execution Loop
    for ds_name in target_datasets:
        data = load_standardized_json(dataset_files[ds_name], max_samples=args.num_samples)
        if not data: 
            continue
            
        print(f"\n--- Evaluating Dataset: {ds_name.upper()} ({len(data)} samples) ---")
        
        for config in target_configs:
            print(f" -> Configuration: {config.upper()}")
            
            master_log = []
            
            for item in tqdm(data, desc=f"Generating ({config})"):
                log_entry = {
                    "dataset": ds_name,
                    "configuration": config,
                    "prompt_id": item.get("id", "unknown"),
                    "expected_domain": item.get("expected_domain", "unknown"),
                    "routed_domain": None,
                    "router_confidence": None,
                    "routing_correct": None,
                    "adapter_switch_ms": 0.0,
                    "generation_ms": 0.0,
                    "tokens_generated": 0,
                    "tps": 0.0,
                    "execution_correct": None,
                    "prompt": item["prompt"],
                    "generated_response": "",
                    "ground_truth_answer": item.get("ground_truth_answer", "")
                }

                target_adapter = "general"
                bypass_lora = False
                
                # --- ROUTING PHASE ---
                if config == "routed":
                    decision, rule, detailed_log, conf = router.predict(
                        item["prompt"], verbose=False, hybrid_margin=0.0, 
                        hybrid_min_confidence=0.30, hybrid_min_existence=0.0
                    )
                    log_entry["routed_domain"] = decision
                    log_entry["router_confidence"] = conf
                    log_entry["routing_correct"] = (decision == item.get("expected_domain"))
                    
                    target_adapter = decision
                    if decision == "general":
                        bypass_lora = True
                else:
                    # Static configs
                    if config == "base":
                        bypass_lora = True
                    elif config == "hybrid_adapter":
                        target_adapter = "hybrid"
                    else:
                        target_adapter = config.replace("pure_", "") # "math" or "code"

                # --- GENERATION PHASE ---
                metrics = llm.generate(
                    user_prompt=item["prompt"],
                    domain=target_adapter,
                    stream=False,
                    bypass_lora=bypass_lora,
                    # max_new_tokens=args.max_new_tokens
                )
                
                log_entry["adapter_switch_ms"] = metrics["switch_ms"]
                log_entry["generation_ms"] = metrics["generation_ms"]
                log_entry["tokens_generated"] = metrics["num_tokens"]
                log_entry["tps"] = metrics["tps"]
                log_entry["generated_response"] = metrics["response"]

                # --- EVALUATION PHASE ---
                truth = item.get("ground_truth_answer")
                expected = item.get("expected_domain")
                
                if truth:
                    if expected == "math" or ds_name == "gsm8k":
                        log_entry["execution_correct"] = evaluate_math(metrics["response"], truth)
                    elif expected == "code" or ds_name == "codefeedback":
                        # Leave as None for offline LLM-as-a-judge evaluation
                        log_entry["execution_correct"] = None 
                    elif expected == "hybrid":
                        # We can evaluate the math portion automatically, but the final
                        # determination requires offline LLM evaluation of the code.
                        # Leaving as None so it isn't accidentally counted as a strict False.
                        log_entry["execution_correct"] = None

                master_log.append(log_entry)

            # Save individual config-dataset combination locally
            out_filename = f"{args.out_dir}/eval_{ds_name}_{config}.json"
            with open(out_filename, "w", encoding="utf-8") as f:
                json.dump(master_log, f, indent=4)
            print(f"    Saved Locally -> {out_filename}")

            # --- BACKUP TO GOOGLE DRIVE ---
            if IN_COLAB:
                drive_filepath = os.path.join(args.drive_dir, f"eval_{ds_name}_{config}.json")
                shutil.copy2(out_filename, drive_filepath)
                print(f"    [✔] Synced to Google Drive -> {drive_filepath}")

    print("\nAll tasks completed successfully.")

if __name__ == "__main__":
    main()