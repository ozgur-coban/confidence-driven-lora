import os
import sys
import json
import time
import shutil
import argparse
import pandas as pd
from collections import defaultdict
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
    
    parser.add_argument("--datasets", nargs="+", choices=["gsm8k", "codefeedback", "mixed", "hybrid", "wildchat", "all"], default=["all"])
    parser.add_argument("--configs", nargs="+", choices=["base", "pure_math", "pure_code", "hybrid_adapter", "routed", "all"], default=["all"])
    parser.add_argument("--code_adapter", default="./models/adapters/adapter_code/code_codefeedback_3b_base_07.06.2026")
    parser.add_argument("--math_adapter", default="./models/adapters/adapter_math/math_gsm8k_main_3b_base_07.06.2026")
    parser.add_argument("--hybrid_adapter", default="./models/adapters/adapter_hybrid/math_code_ties_07.06.2026/hybrid")
    parser.add_argument("--num_samples", type=int, default=-1)
    
    # PARAMETERIZED BATCH SIZE
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for parallel generation.")
    
    parser.add_argument("--out_dir", type=str, default="logs/eval_grid")
    parser.add_argument("--drive_dir", type=str, default="/content/drive/MyDrive/capstone_eval_logs")
    
    return parser.parse_args()

def main():
    args = parse_args()
    load_dotenv()
    
    # --- DETECT COLAB & MOUNT DRIVE ---
    IN_COLAB = "COLAB_RELEASE_TAG" in os.environ
    if IN_COLAB:
        print("\n[!] Google Colab Environment Detected. Mounting Google Drive...")
        from google.colab import drive
        drive.mount('/content/drive')
        os.makedirs(args.drive_dir, exist_ok=True)
        print(f" -> Drive mounted. Logs will be synced to: {args.drive_dir}\n")

    if os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))

    os.makedirs(args.out_dir, exist_ok=True)

    all_datasets = ["gsm8k", "codefeedback", "mixed", "hybrid", "wildchat"]
    all_configs = ["base", "pure_math", "pure_code", "hybrid_adapter", "routed"]
    
    target_datasets = all_datasets if "all" in args.datasets else args.datasets
    target_configs = all_configs if "all" in args.configs else args.configs

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
    print(f"Batch Size: {args.batch_size}")
    print("="*50)

    # 3. Execution Loop
    for ds_name in target_datasets:
        data = load_standardized_json(dataset_files[ds_name], max_samples=args.num_samples)
        if not data: 
            continue
            
        print(f"\n--- Evaluating Dataset: {ds_name.upper()} ({len(data)} samples) ---")
        
        for config in target_configs:
            print(f" -> Configuration: {config.upper()}")
            
            master_log = [None] * len(data)
            routed_tasks = []

            # Routing Pass (instant)
            for i, item in enumerate(data):
                if config == "routed":
                    decision, rule, detailed_log, conf = router.predict(
                        item["prompt"], verbose=False, hybrid_margin=0.0, 
                        hybrid_min_confidence=0.30, hybrid_min_existence=0.0
                    )
                    routed_tasks.append({
                        "idx": i, "item": item, "decision": decision, "conf": conf,
                        "target_adapter": decision, "bypass_lora": (decision == "general")
                    })
                else:
                    target_adapter = "general" if config == "base" else (
                        "hybrid" if config == "hybrid_adapter" else config.replace("pure_", "")
                    )
                    routed_tasks.append({
                        "idx": i, "item": item, "decision": None, "conf": None,
                        "target_adapter": target_adapter, "bypass_lora": (config == "base")
                    })

            # Group for batching
            grouped_tasks = defaultdict(list)
            for task in routed_tasks:
                grouped_tasks[(task["target_adapter"], task["bypass_lora"])].append(task)

            # Generate Pass
            with tqdm(total=len(data), desc=f"Generating ({config})") as pbar:
                for (target_adapter, bypass_lora), tasks in grouped_tasks.items():
                    for i in range(0, len(tasks), args.batch_size):
                        batch = tasks[i : i + args.batch_size]
                        prompts = [t["item"]["prompt"] for t in batch]
                        
                        metrics_list = llm.generate_batch(
                            user_prompts=prompts,
                            domain=target_adapter,
                            bypass_lora=bypass_lora,
                        )
                        
                        for task, metrics in zip(batch, metrics_list):
                            item = task["item"]
                            expected = item.get("expected_domain")
                            
                            log_entry = {
                                "dataset": ds_name,
                                "configuration": config,
                                "prompt_id": item.get("id", "unknown"),
                                "expected_domain": expected,
                                "routed_domain": task["decision"],
                                "router_confidence": task["conf"],
                                "routing_correct": (task["decision"] == expected) if task["decision"] else None,
                                "adapter_switch_ms": metrics["switch_ms"],
                                "generation_ms": metrics["generation_ms"],
                                "tokens_generated": metrics["num_tokens"],
                                "tps": metrics["tps"],
                                "execution_correct": None,
                                "prompt": item["prompt"],
                                "generated_response": metrics["response"],
                                "ground_truth_answer": item.get("ground_truth_answer", "")
                            }

                            # --- EVALUATION PHASE (RESTORED) ---
                            truth = item.get("ground_truth_answer")
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
                            
                            master_log[task["idx"]] = log_entry
                            pbar.update(1)

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