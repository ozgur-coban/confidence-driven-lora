import time
import json
import os
import numpy as np
from datasets import load_dataset
from src.router.syntax_router import SyntaxRouter

def main():
    print("1. Fetching Training Data for Syntax Router...")
    math_ds = load_dataset("openai/gsm8k", "main", split="train")
    code_ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
    med_ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    
    # Stream UltraChat for General Data
    general_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    
    # 6,000 samples per domain
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

    print("\n2. Initializing and Fitting Syntax Router...")
    router = SyntaxRouter()
    
    start_train = time.time()
    router.fit(training_data)
    print(f"Training completed in {time.time() - start_train:.2f} seconds.")
    
    # Ensure artifacts directory exists
    os.makedirs("artifacts", exist_ok=True)
    router.save("artifacts/syntax_router.joblib")

    # --- PRINT THE IDENTIFIERS ---
    print("\n" + "="*50)
    print("TOP 30 IDENTIFIERS (What the model looks for)")
    print("="*50)
    identifiers = router.get_top_identifiers(n=30)
    for domain, features in identifiers.items():
        formatted = ", ".join([f"'{f}'" for f in features])
        print(f"{domain.upper():<8} -> {formatted}")

    print("\n" + "="*50)
    print("3. ACCURACY & ERROR ANALYSIS")
    print("="*50)

    # General Test Data 
    general_test_prompts = []
    needed = 500
    skip = 6000
    curr = 0
    stream = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    for row in stream:
        messages = row['messages']
        if len(messages) > 0 and messages[0]['role'] == 'user':
            if curr >= skip:
                general_test_prompts.append(messages[0]['content'])
            curr += 1  
        if len(general_test_prompts) >= needed:
            break

    test_data = {
        "math": [item['question'] for item in math_ds.select(range(6000, 6500))],
        "code": [item['query'] for item in code_ds.select(range(6000, 6500))],
        "medical": [item['question'] for item in med_ds.select(range(6000, 6500))],
        "general": general_test_prompts
    }

    domain_stats = {}
    detailed_errors = []  # List to store full error details

    for true_domain, prompts in test_data.items():
        print(f"--- Testing {true_domain.upper()} Prompts ---")
        scores = []
        correct_count = 0
        misclassifications = []
        
        for prompt in prompts:
            predicted_domain, score = router.predict_with_score(prompt)
            scores.append(score)
            
            if predicted_domain == true_domain:
                correct_count += 1
            else:
                # Track for console output
                misclassifications.append((predicted_domain, score))
                
                # Track for JSON export
                detailed_errors.append({
                    "true_label": true_domain,
                    "predicted_label": predicted_domain,
                    "confidence": float(score), # float cast for JSON serialization
                    "prompt": prompt
                })
                    
        domain_stats[true_domain] = {
            "min": min(scores),
            "max": max(scores),
            "avg": sum(scores) / len(scores),
            "accuracy": (correct_count / len(prompts)) * 100,
            "errors": misclassifications,
            "raw_scores": np.array(scores)
        }

    # --- SAVE ERRORS TO JSON ---
    error_file_path = "artifacts/syntax_router_errors.json"
    with open(error_file_path, "w", encoding="utf-8") as f:
        json.dump(detailed_errors, f, indent=4, ensure_ascii=False)
    print(f"\n[INFO] Saved {len(detailed_errors)} misclassified samples to {error_file_path}")

    # --- FINAL METRICS ---
    print("\n" + "="*50)
    print("FINAL ROUTING METRICS")
    print("="*50)
    for domain, stats in domain_stats.items():
        print(f"{domain.upper():<10} -> Accuracy: {stats['accuracy']:>5.1f}% | Min: {stats['min']:.3f} | Max: {stats['max']:.3f} | Avg: {stats['avg']:.3f}")
        if stats['errors']:
            error_counts = {}
            for err_domain, _ in stats['errors']:
                error_counts[err_domain] = error_counts.get(err_domain, 0) + 1
            print(f"           -> Misclassified as: {error_counts}")

    # --- PERCENTILES ---
    print("\n" + "="*50)
    print("SCORE DISTRIBUTIONS (Percentiles)")
    print("="*50)
    for domain, stats in domain_stats.items():
        scores = stats["raw_scores"]
        p5, p25, p50, p75, p95 = np.percentile(scores, [5, 25, 50, 75, 95])
        print(f"{domain.upper():<10} -> 5th: {p5:.3f} | 25th: {p25:.3f} | Median: {p50:.3f} | 75th: {p75:.3f} | 95th: {p95:.3f}")

    # --- THRESHOLD SWEEP ---
    print("\n" + "="*50)
    print("THRESHOLD SWEEP SIMULATION")
    print("="*50)
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    
    print(f"{'Threshold':<10} | {'Gen (Correct)':<18} | {'Math (Blocked)':<18} | {'Code (Blocked)':<18} | {'Med (Blocked)':<18}")
    print("-" * 90)
    
    for t in thresholds:
        gen_acc = (domain_stats["general"]["raw_scores"] >= t).mean() * 100
        math_fn = (domain_stats["math"]["raw_scores"] < t).mean() * 100
        code_fn = (domain_stats["code"]["raw_scores"] < t).mean() * 100
        med_fn = (domain_stats["medical"]["raw_scores"] < t).mean() * 100
        
        print(f"{t:<10.2f} | {gen_acc:>5.1f}% correct     | {math_fn:>5.1f}% blocked    | {code_fn:>5.1f}% blocked    | {med_fn:>5.1f}% blocked")

if __name__ == "__main__":
    main()