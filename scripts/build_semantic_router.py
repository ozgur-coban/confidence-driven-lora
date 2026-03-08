import time
import json
import os
import argparse
import numpy as np
from datasets import load_dataset
from src.router.semantic_router import SemanticRouter
def parse_args():
    parser = argparse.ArgumentParser(description="Train and/or Evaluate Semantic Router.")
    parser.add_argument("--enable_training", action="store_true", help="If flag is passed, fits new centroids and saves them. Otherwise, loads from disk.")
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.35, 
        help="Score threshold for routing (default: 0.35)"
    )
    return parser.parse_args()
def main():
    args=parse_args()
    print("1. Fetching Training Data (Specialists + UltraChat)...")
    math_ds = load_dataset("openai/gsm8k", "main", split="train")
    code_ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
    med_ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    
    # Use UltraChat for the General centroid
    general_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    router = SemanticRouter()
    if args.enable_training:
        general_prompts = []
        for row in general_ds:
            messages = row['messages']
            if len(messages) > 0 and messages[0]['role'] == 'user':
                general_prompts.append(messages[0]['content'])
            if len(general_prompts) >= 6000:
                break

        training_data = {
            "math": [item['question'] for item in math_ds.select(range(6000))],
            "code": [item['query'] for item in code_ds.select(range(6000))],
            "medical": [item['question'] for item in med_ds.select(range(6000))],
            "general": general_prompts
        }

        print("\n2. Initializing and Fitting Semantic Router...")
        
        start_time = time.time()
        router.fit(training_data)
        print(f"Centroids calculated in {time.time() - start_time:.2f}s")
        
        os.makedirs("artifacts", exist_ok=True)
        router.save("artifacts/semantic_centroids.npy")
    else:
        print("\n1. Training disabled. Loading pre-trained Semantic Router...")
        router.load("artifacts/semantic_centroids.npy")

    print("\n" + "="*50)
    print("3. ACCURACY & ERROR ANALYSIS")
    print("="*50)

    # Prepare Test Data (Unseen samples)
    general_test_prompts = []
    stream = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    curr = 0
    for row in stream:
        # Skip the training window
        if curr >= 6000:
            messages = row['messages']
            if len(messages) > 0 and messages[0]['role'] == 'user':
                general_test_prompts.append(messages[0]['content'])
        curr += 1
        if len(general_test_prompts) >= 500:
            break

    test_data = {
        "math": [item['question'] for item in math_ds.select(range(6000, 6500))],
        "code": [item['query'] for item in code_ds.select(range(6000, 6500))],
        "medical": [item['question'] for item in med_ds.select(range(6000, 6500))],
        "general": general_test_prompts
    }

    domain_stats = {}
    detailed_errors = []

    for true_domain, prompts in test_data.items():
        print(f"--- Testing {true_domain.upper()} Prompts ---")
        scores = []
        correct_count = 0
        forced_to_general_count = 0
        misclassifications = []
        
        for prompt in prompts:
            final_decision, best_guess, score = router.predict(prompt, threshold=args.threshold)
            scores.append(score)
            
            # Track if it was a specialty domain that got downgraded due to low confidence
            if final_decision == "general" and best_guess != "general" and score < args.threshold:
                forced_to_general_count += 1

            if final_decision == true_domain:
                correct_count += 1
            else:
                misclassifications.append((final_decision, score))
                detailed_errors.append({
                    "true_label": true_domain,
                    "final_decision": final_decision,
                    "router_best_guess": best_guess,
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

    # Save detailed error log to JSON
    error_file = "artifacts/semantic_router_errors.json"
    os.makedirs(os.path.dirname(error_file), exist_ok=True)
    with open(error_file, "w", encoding="utf-8") as f:
        json.dump(detailed_errors, f, indent=4, ensure_ascii=False)

    # FINAL REPORTING
    print("\n" + "="*60)
    print(f"FINAL ROUTING METRICS (With {args.threshold} Confidence Threshold)")
    print("="*60)
    for domain, stats in domain_stats.items():
        print(f"[{domain.upper()}]")
        print(f"  -> Accuracy: {stats['accuracy']:>5.1f}%")
        print(f"  -> Avg Confidence Score: {stats['avg']:.3f}")
        if domain != "general":
            print(f"  -> Downgraded to General (Score < 0.35): {stats['forced_to_general']:>5.1f}%")
        
        if stats['errors']:
             err_counts = {}
             for d, _ in stats['errors']: 
                 err_counts[d] = err_counts.get(d, 0) + 1
             print(f"  -> Misclassified As: {err_counts}")
        print("-" * 60)

    print("\n" + "="*60)
    print("THRESHOLD SWEEP (Impact of raising the gatekeeper threshold)")
    print("="*60)
    print("If threshold is X, what % of specialty prompts get downgraded to General?")
    print(f"{'Threshold':<10} | {'Math Downgraded':<18} | {'Code Downgraded':<18} | {'Med Downgraded':<18}")
    print("-" * 75)
    for t in [0.15, 0.25, 0.30, 0.35, 0.40, 0.45,0.50]:
        math_down = (domain_stats["math"]["raw_scores"] < t).mean() * 100
        code_down = (domain_stats["code"]["raw_scores"] < t).mean() * 100
        med_down = (domain_stats["medical"]["raw_scores"] < t).mean() * 100
        print(f"{t:.2f}       | {math_down:>5.1f}%             | {code_down:>5.1f}%             | {med_down:>5.1f}%")

if __name__ == "__main__":
    main()