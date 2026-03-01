import time
import numpy as np
from datasets import load_dataset
from src.router.semantic_router import SemanticRouter

def main():
    print("1. Fetching Training Data for Centroids...")
    math_ds = load_dataset("openai/gsm8k", "main", split="train")
    code_ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
    med_ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    
    # We add a General dataset just for testing the threshold (not for training centroids)
    general_ds = load_dataset("tatsu-lab/alpaca", split="train")

    training_data = {
        "math": [item['question'] for item in math_ds.select(range(6000))],
        "code": [item['query'] for item in code_ds.select(range(6000))],
        "medical": [item['question'] for item in med_ds.select(range(6000))]
    }

    print("\n2. Initializing and Fitting Semantic Router...")
    router = SemanticRouter()
    router.fit(training_data)
    router.save("artifacts/router_centroids.npy")

    print("\n" + "="*50)
    print("3. ACCURACY & THRESHOLD ANALYSIS")
    print("="*50)

    # 500 unseen samples for each domain to analyze scores and accuracy
    test_data = {
        "math": [item['question'] for item in math_ds.select(range(6000, 6500))],
        "code": [item['query'] for item in code_ds.select(range(6000, 6500))],
        "medical": [item['question'] for item in med_ds.select(range(6000, 6500))],
        "general": [item['instruction'] for item in general_ds.select(range(500))]
    }

    domain_stats = {}

    for true_domain, prompts in test_data.items():
        print(f"\n--- Testing {true_domain.upper()} Prompts ---")
        scores = []
        correct_count = 0
        misclassifications = []
        
        for prompt in prompts:
            predicted_domain, score = router.predict_with_score(prompt)
            scores.append(score)
            
            # For general, 'correct' means it should score LOW (we'll handle threshold later)
            # For math/code/med, 'correct' means predicted_domain == true_domain
            if true_domain != "general":
                if predicted_domain == true_domain:
                    correct_count += 1
                else:
                    misclassifications.append((predicted_domain, score))
                    
        domain_stats[true_domain] = {
            "min": min(scores),
            "max": max(scores),
            "avg": sum(scores) / len(scores),
            "accuracy": (correct_count / len(prompts)) * 100 if true_domain != "general" else None,
            "errors": misclassifications,
            "raw_scores": np.array(scores)  # Stored for percentile/sweep calculations
        }

    # ---------------------------------------------------------
    # PART A: ORIGINAL MIN/MAX/AVG & MISCLASSIFICATIONS
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("FINAL ROUTING METRICS")
    print("="*50)
    for domain, stats in domain_stats.items():
        if domain == "general":
            print(f"GENERAL    -> Min: {stats['min']:.3f} | Max: {stats['max']:.3f} | Avg: {stats['avg']:.3f}")
        else:
            print(f"{domain.upper():<10} -> Accuracy: {stats['accuracy']:>5.1f}% | Min: {stats['min']:.3f} | Max: {stats['max']:.3f} | Avg: {stats['avg']:.3f}")
            if stats['errors']:
                error_counts = {}
                for err_domain, _ in stats['errors']:
                    error_counts[err_domain] = error_counts.get(err_domain, 0) + 1
                print(f"           -> Misclassified as: {error_counts}")

    # ---------------------------------------------------------
    # PART B: PERCENTILE DISTRIBUTIONS
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("SCORE DISTRIBUTIONS (Percentiles)")
    print("="*50)
    for domain, stats in domain_stats.items():
        scores = stats["raw_scores"]
        p5, p25, p50, p75, p95 = np.percentile(scores, [5, 25, 50, 75, 95])
        print(f"{domain.upper():<10} -> 5th: {p5:.3f} | 25th: {p25:.3f} | Median: {p50:.3f} | 75th: {p75:.3f} | 95th: {p95:.3f}")

    # ---------------------------------------------------------
    # PART C: THRESHOLD SWEEP SIMULATION
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("THRESHOLD SWEEP SIMULATION")
    print("="*50)
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    
    print(f"{'Threshold':<10} | {'Gen (False Pos)':<18} | {'Math (Blocked)':<18} | {'Code (Blocked)':<18} | {'Med (Blocked)':<18}")
    print("-" * 90)
    
    for t in thresholds:
        # False Positives: General prompts that scored higher than threshold (Bad)
        gen_fp = (domain_stats["general"]["raw_scores"] >= t).mean() * 100
        
        # False Negatives: Valid prompts that scored lower than threshold (Bad)
        math_fn = (domain_stats["math"]["raw_scores"] < t).mean() * 100
        code_fn = (domain_stats["code"]["raw_scores"] < t).mean() * 100
        med_fn = (domain_stats["medical"]["raw_scores"] < t).mean() * 100
        
        print(f"{t:<10.2f} | {gen_fp:>5.1f}% let through | {math_fn:>5.1f}% blocked    | {code_fn:>5.1f}% blocked    | {med_fn:>5.1f}% blocked")

if __name__ == "__main__":
    main()