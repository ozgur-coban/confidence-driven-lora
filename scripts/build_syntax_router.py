import time
import json
import os
import argparse
import numpy as np
from src.router.syntax_router import SyntaxRouter
from src.data_ops.dataset_manager import DatasetManager

def parse_args():
    parser = argparse.ArgumentParser(description="Train and/or Evaluate Syntax Router.")
    parser.add_argument("--enable_training", action="store_true", help="Fit new vectorizer/classifier and save.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    router = SyntaxRouter()
    data_manager = DatasetManager()

    if args.enable_training:
        print("\n1. Fetching Training Data from Data Manager...")
        # You can tweak the General mixture here. 
        # 50k total (25k SQuAD + 25k UltraChat) is a massive, balanced background.
        training_data = data_manager.get_training_data(
            specialist_samples=6000, 
            squad_samples=100000, 
            ultra_samples=150000
        )

        print("\n2. Initializing and Fitting Syntax Router...")
        start_train = time.time()
        router.fit(training_data)
        print(f"Training completed in {time.time() - start_train:.2f} seconds.")
        
        os.makedirs("artifacts", exist_ok=True)
        router.save("./models/syntax_router.joblib")
        
        print("\n" + "="*50)
        print("TOP 15 IDENTIFIERS (What the model looks for)")
        print("="*50)
        identifiers = router.get_top_identifiers(n=15)
        for domain, features in identifiers.items():
            formatted = ", ".join([f"'{f}'" for f in features])
            print(f"{domain.upper():<8} -> {formatted}")
    else:
        print("\n1. Training disabled. Loading pre-trained Syntax Router...")
        router.load("./models/syntax_router.joblib")

    print("\n" + "="*50)
    print("3. ACCURACY & ERROR ANALYSIS")
    print("="*50)

    # Clean, one-line test data loading
    test_data = data_manager.get_testing_data(test_samples=500)

    domain_stats = {}
    detailed_errors = []

    for true_domain, prompts in test_data.items():
        print(f"--- Testing {true_domain.upper()} Prompts ---")
        scores = []
        correct_count = 0
        forced_to_general_count = 0
        misclassifications = []
        
        for prompt in prompts:
            predicted_domain, score = router.predict_with_score(prompt)
            scores.append(score)
            
            # Using 0.50 threshold gatekeeper
            final_decision = predicted_domain if score >= 0.50 else "general"
            
            if final_decision == "general" and predicted_domain != "general" and score < 0.50:
                forced_to_general_count += 1

            if final_decision == true_domain:
                correct_count += 1
            else:
                misclassifications.append((final_decision, score))
                detailed_errors.append({
                    "true_label": true_domain,
                    "final_decision": final_decision,
                    "router_best_guess": predicted_domain,
                    "confidence": float(score),
                    "prompt": prompt
                })
                    
        domain_stats[true_domain] = {
            "min": min(scores), "max": max(scores), "avg": sum(scores) / len(scores),
            "accuracy": (correct_count / len(prompts)) * 100,
            "forced_to_general": (forced_to_general_count / len(prompts)) * 100,
            "errors": misclassifications,
            "raw_scores": np.array(scores)
        }

    error_file_path = "./logs/syntax_router_errors.json"
    with open(error_file_path, "w", encoding="utf-8") as f:
        json.dump(detailed_errors, f, indent=4, ensure_ascii=False)

    print("\n" + "="*60)
    print("FINAL ROUTING METRICS (With 0.50 Confidence Threshold)")
    print("="*60)
    for domain, stats in domain_stats.items():
        print(f"[{domain.upper()}]")
        print(f"  -> Accuracy: {stats['accuracy']:>5.1f}%")
        print(f"  -> Avg Confidence Score: {stats['avg']:.3f}")
        if domain != "general":
            print(f"  -> Downgraded to General (Score < 0.50): {stats['forced_to_general']:>5.1f}%")
        
        if stats['errors']:
             err_counts = {}
             for d, _ in stats['errors']: 
                 err_counts[d] = err_counts.get(d, 0) + 1
             print(f"  -> Misclassified As: {err_counts}")
        print("-" * 60)

    print("\n" + "="*60)
    print("THRESHOLD SWEEP (Impact of raising the gatekeeper threshold)")
    print("="*60)
    print(f"{'Threshold':<10} | {'Math Downgraded':<18} | {'Code Downgraded':<18} | {'Med Downgraded':<18}")
    print("-" * 75)
    for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
        math_down = (domain_stats["math"]["raw_scores"] < t).mean() * 100
        code_down = (domain_stats["code"]["raw_scores"] < t).mean() * 100
        med_down = (domain_stats["medical"]["raw_scores"] < t).mean() * 100
        print(f"{t:.2f}       | {math_down:>5.1f}%             | {code_down:>5.1f}%             | {med_down:>5.1f}%")

if __name__ == "__main__":
    main()