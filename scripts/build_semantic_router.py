import time
import json
import os
import numpy as np
from datasets import load_dataset
from src.router.semantic_router import SemanticRouter

def main():
    print("1. Fetching Training Data (Specialists + UltraChat)...")
    math_ds = load_dataset("openai/gsm8k", "main", split="train")
    code_ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
    med_ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    
    # Use UltraChat for the General centroid
    general_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    
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
    router = SemanticRouter()
    start_time = time.time()
    router.fit(training_data)
    print(f"Centroids calculated in {time.time() - start_time:.2f}s")
    
    os.makedirs("artifacts", exist_ok=True)
    router.save("artifacts/semantic_centroids.npy")

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
        misclassifications = []
        unknown_count=0
        
        for prompt in prompts:
            # We use predict_with_score to see the 'raw' choice before thresholds
            predicted_domain, score = router.predict(prompt,threshold=0.2)
            scores.append(score)
            
            if predicted_domain == true_domain:
                correct_count += 1
            elif predicted_domain == "unknown":
                unknown_count += 1
                detailed_errors.append({
                    "true_label": true_domain,
                    "predicted_label": "unknown",
                    "confidence": float(score),
                    "prompt": prompt
                })
            else:
                misclassifications.append((predicted_domain, score))
                detailed_errors.append({
                    "true_label": true_domain,
                    "predicted_label": predicted_domain,
                    "confidence": float(score),
                    "prompt": prompt
                })
                    
        domain_stats[true_domain] = {
            "min": min(scores), "max": max(scores), "avg": sum(scores) / len(scores),
            "accuracy": (correct_count / len(prompts)) * 100,
            "unknown_rate": (unknown_count / len(prompts)) * 100,
            "errors": misclassifications,
            "raw_scores": np.array(scores)
        }

    # Save detailed error log to JSON
    error_file = "artifacts/semantic_router_errors.json"
    with open(error_file, "w", encoding="utf-8") as f:
        json.dump(detailed_errors, f, indent=4, ensure_ascii=False)

    # FINAL REPORTING
    print("\n" + "="*50)
    print("FINAL ROUTING METRICS (With 0.2 Threshold)")
    print("="*50)
    for domain, stats in domain_stats.items():
        print(f"{domain.upper():<10} -> Accuracy: {stats['accuracy']:>5.1f}% | Unknown: {stats['unknown_rate']:>5.1f}% | Avg Score: {stats['avg']:.3f}")
        if stats['errors']:
             err_counts = {}
             for d, _ in stats['errors']: err_counts[d] = err_counts.get(d, 0) + 1
             print(f"           -> Misclassified as: {err_counts}")

    print("\n" + "="*50)
    print("SCORE DISTRIBUTIONS (Percentiles)")
    print("="*50)
    for domain, stats in domain_stats.items():
        p5, p50, p95 = np.percentile(stats["raw_scores"], [5, 50, 95])
        print(f"{domain.upper():<10} -> 5th: {p5:.3f} | Median: {p50:.3f} | 95th: {p95:.3f}")

    print("\n" + "="*50)
    print("THRESHOLD SWEEP SIMULATION")
    print("="*50)
    print(f"{'Threshold':<10} | {'Gen (Correct)':<18} | {'Math (Blocked)':<18} | {'Code (Blocked)':<18}")
    print("-" * 75)
    for t in [0.25, 0.30, 0.35, 0.40, 0.45]:
        gen_correct = (domain_stats["general"]["raw_scores"] >= t).mean() * 100
        math_blocked = (domain_stats["math"]["raw_scores"] < t).mean() * 100
        code_blocked = (domain_stats["code"]["raw_scores"] < t).mean() * 100
        print(f"{t:.2f}      | {gen_correct:>5.1f}% correct     | {math_blocked:>5.1f}% blocked    | {code_blocked:>5.1f}% blocked")

if __name__ == "__main__":
    main()