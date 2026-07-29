import os
import json
import sys
from src.router.meta_router import MetaRouter

def run_wildchat_eval(wildchat_path="./data/wildchat/hybrid_wildchat_from_gemini_filtered_further.json"):
    # 1. Initialize and load the frozen router
    router = MetaRouter()
    
    print("Loading pre-trained Meta Router...")
    router.load(
        semantic_path="models/semantic_centroids.npy",
        syntax_path="models/syntax_router.joblib",
        meta_path="models/meta_router.joblib"
    )
    
    # 2. Load the 150 WildChat prompts
    if not os.path.exists(wildchat_path):
        raise FileNotFoundError(f"Could not find WildChat test file at {wildchat_path}")
        
    with open(wildchat_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    prompts = []
    for item in data:
        if isinstance(item, dict) and "prompt" in item:
            prompts.append(item["prompt"])
        elif isinstance(item, str):
            prompts.append(item)
    
    print(f"\nLoaded {len(prompts)} OOD prompts from {wildchat_path}")
    
    # 3. Track metrics (Ground truth is assumed to be 'hybrid' for this test set)
    results = {
        "math": 0,
        "code": 0,
        "hybrid": 0,
        "general": 0
    }
    
    detailed_errors = []
    correct_classifications = []
    
    # 4. Predict using your highly optimized parameters
    print("Running inference...")
    for prompt in prompts:
        decision, rule, detailed_log = router.predict(
            prompt, 
            verbose=False,
            hybrid_margin=0.05,
            hybrid_min_confidence=0.38,
            hybrid_min_existence=0.001
        )
        
        results[decision] += 1
        
        log_entry = {
            "true_domain": "hybrid",
            "predicted_domain": decision,
            "rule_triggered": rule,
            "prompt": prompt,
            "router_internal_log": detailed_log.strip()
        }
        
        if decision == "hybrid":
            correct_classifications.append(log_entry)
        else:
            detailed_errors.append(log_entry)
            
    # 5. Print the Routing Distribution
    print("\n" + "="*45)
    print("WILDCHAT OOD EVALUATION RESULTS (Expected: HYBRID)")
    print("="*45)
    for domain, count in results.items():
        percentage = (count / len(prompts)) * 100 if len(prompts) > 0 else 0
        print(f"{domain:>10} : {count:>3} ({percentage:.1f}%)")
        
    accuracy = (results["hybrid"] / len(prompts)) * 100 if len(prompts) > 0 else 0
    print("-" * 45)
    print(f"OOD Hybrid Recall: {accuracy:.1f}%")
        
    # 6. Save the detailed error logs for manual inspection
    error_out_path = "./logs/wildchat_meta_router_errors.json"
    os.makedirs(os.path.dirname(error_out_path), exist_ok=True)
    with open(error_out_path, "w", encoding="utf-8") as f:
        json.dump(detailed_errors, f, indent=4, ensure_ascii=False)
        
    print("="*45)
    print(f"Saved {len(detailed_errors)} detailed misclassification logs to '{error_out_path}' for review.")

if __name__ == "__main__":
    run_wildchat_eval(wildchat_path="./data/wildchat/hybrid_wildchat_from_gemini_filtered_further.json")