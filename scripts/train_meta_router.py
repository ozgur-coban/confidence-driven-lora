import sys
import os
import time
import json
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_ops.routing_dataset_manager import DatasetManager  
from src.router.meta_router import MetaRouter
from config.router_dataset_configs import ROUTER_DATASET_CONFIGS

def parse_args():
    parser = argparse.ArgumentParser(description="Train and/or Evaluate Meta Router.")
    parser.add_argument(
        "--enable_training",
        action="store_true",
        help="If flag is passed, fits a new meta-classifier and saves it. Otherwise, loads from disk.",
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
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Initialize the unified Dataset Manager
    local_paths = ROUTER_DATASET_CONFIGS.get("local_paths")
    domain_rules = ROUTER_DATASET_CONFIGS.get("domain_rules")
    if "medical" in domain_rules:
        del domain_rules["medical"]

    data_manager = DatasetManager(
        local_paths=local_paths, domain_configs=domain_rules
    )

    router = MetaRouter()

    # ==========================================
    # PHASE 1: TRAINING OR LOADING
    # ==========================================
    if args.enable_training:
        print(f"\n1. Fetching Training Data ({args.specialist_samples} samples per domain)...")
        
        # Pull balanced datasets
        training_data = data_manager.get_training_data(
            specialist_samples=args.specialist_samples,
            general_samples=0,
            squad_samples=int(args.specialist_samples / 2),
            ultra_samples=int(args.specialist_samples / 2),
        )

        print("\n2. Initializing and Fitting Meta Router...")
        # Load the base sub-routers first so they can extract features
        router.load(
            semantic_path="models/semantic_centroids.npy",
            syntax_path="models/syntax_router.joblib",
            meta_path="models/meta_router.joblib" # Will safely warn it doesn't exist yet
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
    print("3. ACCURACY & ERROR ANALYSIS (Holdout Set)")
    print("=" * 70)

    # Fetch holdout data by offsetting the start index by the number of training samples
    test_data = data_manager.get_testing_data(
        test_samples=args.test_samples, 
        specialist_offset=args.specialist_samples
    )

    domain_labels = ["math", "code", "hybrid", "general"]
    
    all_y_true = []
    all_y_pred = []
    detailed_errors = []
    
    total_correct = 0
    total_samples = 0

    for true_domain, prompts in test_data.items():
        if true_domain not in domain_labels:
            continue

        print(f"--- Testing {true_domain.upper()} Prompts ---")
        correct = 0
        rule_triggers = {}
        
        start_time = time.time()
        
        for prompt in prompts:
            # Pass verbose=False and log_file=None to keep terminal clean during calculation
            final_decision, logic_rule, detailed_log = router.predict(
                prompt, verbose=False, log_file=None
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

        # Print domain-specific profiling metrics
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

    # Highly scannable confusion matrix mapping
    dummy_title="'True \ Pred':>15"
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