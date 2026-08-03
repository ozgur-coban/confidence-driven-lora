import sys
import os
import time
import json
import argparse
import numpy as np
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_ops.routing_dataset_manager import DatasetManager  
from src.router.meta_router import MetaRouter
from config.router_dataset_configs import ROUTER_DATASET_CONFIGS

def parse_args():
    parser = argparse.ArgumentParser(description="Train and Evaluate Meta Router with Verbose Diagnostics.")
    parser.add_argument(
        "--enable_training",
        action="store_true",
        help="Fits a new meta-classifier and saves it. Otherwise loads from disk.",
    )
    parser.add_argument(
        "--specialist_samples",
        type=int,
        default=240,
        help="Number of samples per domain to use for training.",
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=60,
        help="Number of unseen holdout samples to evaluate.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print raw feature vectors and prediction distributions for every evaluated prompt.",
    )
    return parser.parse_args()

def load_combined_hybrid_data(hybrid_cfg, sample_count, is_test=False):
    """Combines Hugging Face hybrid dataset with local GSM8K synthetic hybrid dataset."""
    prompts = []
    
    # 1. Load HF Hybrid Dataset (300 samples)
    try:
        split = hybrid_cfg["test_split"] if is_test else hybrid_cfg["train_split"]
        hf_ds = load_dataset(hybrid_cfg["path"], split=split)
        col = hybrid_cfg["column"]
        hf_prompts = [row[col] for row in hf_ds if row.get(col)]
        prompts.extend(hf_prompts)
        print(f" -> Loaded {len(hf_prompts)} hybrid samples from HF '{hybrid_cfg['path']}'")
    except Exception as e:
        print(f" -> Warning: Failed to load HF hybrid dataset: {e}")

    # 2. Load Local Synthetic GSM8K Hybrid Dataset (~1050 samples)
    local_key = "local_test" if is_test else "local_train"
    local_path = hybrid_cfg.get(local_key)
    if local_path and os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                local_prompts = json.load(f)
                prompts.extend(local_prompts)
                print(f" -> Loaded {len(local_prompts)} hybrid samples from local file '{local_path}'")
        except Exception as e:
            print(f" -> Warning: Failed to load local hybrid file '{local_path}': {e}")
            
    if not prompts:
        raise ValueError("No hybrid data could be loaded. Check file paths and network connection.")

    # Shuffle and trim to target sample_count
    np.random.seed(42)
    np.random.shuffle(prompts)
    
    if len(prompts) < sample_count:
        print(f" -> Warning: Requested {sample_count} samples, but only have {len(prompts)} total. Using all available.")
        return prompts
        
    return prompts[:sample_count]

def main():
    args = parse_args()

    local_paths = ROUTER_DATASET_CONFIGS.get("local_paths")
    # Make a copy so we don't mutate the original dictionary
    domain_rules = ROUTER_DATASET_CONFIGS.get("domain_rules").copy() 
    
    if "medical" in domain_rules:
        del domain_rules["medical"]

    # --- THE FIX: Extract hybrid configuration and remove it from DatasetManager's queue ---
    hybrid_cfg = domain_rules.pop("hybrid")

    # DatasetManager will now ONLY load math, code, and general
    data_manager = DatasetManager(local_paths=local_paths, domain_configs=domain_rules)
    router = MetaRouter()

    # ==========================================
    # PHASE 1: TRAINING OR LOADING
    # ==========================================
    if args.enable_training:
        print(f"\n1. Fetching Training Data ({args.specialist_samples} samples per domain)...")
        
        training_data = data_manager.get_training_data(
            specialist_samples=args.specialist_samples,
            general_samples=0,
            squad_samples=int(args.specialist_samples / 2),
            ultra_samples=int(args.specialist_samples / 2),
        )

        # Manually load the combined hybrid data
        training_data["hybrid"] = load_combined_hybrid_data(
            hybrid_cfg, sample_count=args.specialist_samples, is_test=False
        )

        print("\n2. Initializing and Fitting Meta Router...")
        router.load(
            semantic_path="models/semantic_centroids.npy",
            syntax_path="models/syntax_router.joblib",
            meta_path="models/meta_router.joblib"
        )
        
        start_time = time.time()
        router.fit(training_data, save_path="models/meta_router.joblib")
        print(f"Meta-Classifier trained and saved in {time.time() - start_time:.2f}s")

    else:
        print("\n1. Training disabled. Loading pre-trained Meta Router...")
        router.load(
            semantic_path="models/semantic_centroids.npy",
            syntax_path="models/syntax_router.joblib",
            meta_path="models/meta_router.joblib"
        )

    # ==========================================
    # PHASE 2: ACCURACY & ERROR ANALYSIS
    # ==========================================
    print("\n" + "=" * 70)
    print("3. ACCURACY & ERROR ANALYSIS (Combined Holdout Set)")
    print("=" * 70)

    test_data = data_manager.get_testing_data(
        test_samples=args.test_samples, 
        specialist_offset=args.specialist_samples
    )

    # Manually load the combined hybrid test data
    test_data["hybrid"] = load_combined_hybrid_data(
        hybrid_cfg, sample_count=args.test_samples, is_test=True
    )

    domain_labels = ["math", "code", "hybrid", "general"]
    
    all_y_true = []
    all_y_pred = []
    detailed_errors = []
    
    total_correct = 0
    total_samples = 0

    for true_domain in domain_labels:
        prompts = test_data.get(true_domain, [])
        if not prompts:
            continue

        print(f"\n--- Testing {true_domain.upper()} Prompts ({len(prompts)} samples) ---")
        correct = 0
        rule_triggers = {}
        
        start_time = time.time()
        
        for prompt in prompts:
            final_decision, logic_rule, detailed_log,_ = router.predict(
                prompt, 
                verbose=args.verbose, 
                log_file="logs/meta_router_eval_run.log",
                hybrid_margin=0.05,             # Changed from 0.15
                hybrid_min_confidence=0.38,     # Changed from 0.50
                hybrid_min_existence=0.001       # Changed from 0.05
            )

            all_y_true.append(true_domain)
            all_y_pred.append(final_decision)
            
            rule_triggers[logic_rule] = rule_triggers.get(logic_rule, 0) + 1

            if final_decision == true_domain:
                correct += 1
            else:
                detailed_errors.append({
                    "true_domain": true_domain,
                    "predicted_domain": final_decision,
                    "rule_triggered": logic_rule,
                    "prompt": prompt,
                    "router_internal_log": detailed_log.strip(),
                })

        elapsed = time.time() - start_time
        accuracy = (correct / len(prompts)) * 100 if len(prompts) > 0 else 0.0
        
        total_correct += correct
        total_samples += len(prompts)

        print(f"  -> Accuracy: {accuracy:>5.1f}%")
        print(f"  -> Speed:    {len(prompts) / elapsed if elapsed > 0 else 0:.1f} prompts/sec")
        print(f"  -> Handled by:")
        for rule, count in sorted(rule_triggers.items(), key=lambda x: x[1], reverse=True):
            print(f"       - {count:>3} : {rule}")
        print("-" * 70)

    # Save detailed error log to JSON
    error_file_path = "./logs/meta_router_errors.json"
    os.makedirs(os.path.dirname(error_file_path), exist_ok=True)
    with open(error_file_path, "w", encoding="utf-8") as f:
        json.dump(detailed_errors, f, indent=4, ensure_ascii=False)

    # ==========================================
    # PHASE 3: ADVANCED METRICS 
    # ==========================================
    print("\n" + "=" * 70)
    print("ROUTER CONFUSION MATRIX")
    print("=" * 70)
    cm = confusion_matrix(all_y_true, all_y_pred, labels=domain_labels)

    dummy_title = f"{'True || Pred':>15}"
    header = f"{dummy_title} |" + "".join([f"{label:>10}" for label in domain_labels])
    print(header)
    print("-" * len(header))
    for i, true_label in enumerate(domain_labels):
        row_str = f"{true_label:>15} |" + "".join([f"{val:>10}" for val in cm[i]])
        print(row_str)

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT (Precision, Recall, F1)")
    print("=" * 70)
    report = classification_report(
        all_y_true, all_y_pred, labels=domain_labels, target_names=domain_labels, zero_division=0
    )
    print(report)

    overall_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
    print(f"\nOVERALL HYBRID ACCURACY: {overall_accuracy:.2f}%")
    print(f"Detailed error logs saved to: {error_file_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()