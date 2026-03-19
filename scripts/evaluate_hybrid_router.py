import os
import json
import time
from datasets import load_dataset
from src.router.hybrid_router import HybridRouter

def main():
    print("1. Initializing Hybrid Environment...")
    router = HybridRouter()
    router.load()
    
    print("\n2. Fetching Exact Test Data (Unseen samples)...")
    math_ds = load_dataset("openai/gsm8k", "main", split="train")
    code_ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
    med_ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")

    general_test_prompts = []
    stream = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft", streaming=True)
    for row in stream:
        messages = row['messages']
        if len(messages) > 0 and messages[0]['role'] == 'user':
            general_test_prompts.append(messages[0]['content'])
        if len(general_test_prompts) >= 500:
            break

    test_data = {
        "math": [item['question'] for item in math_ds.select(range(6000, 6500))],
        "code": [item['query'] for item in code_ds.select(range(6000, 6500))],
        "medical": [item['question'] for item in med_ds.select(range(6000, 6500))],
        "general": general_test_prompts
    }

    print("\n" + "="*70)
    print("HYBRID ROUTER ACCURACY REPORT")
    print("="*70)

    total_correct = 0
    total_samples = 0
    detailed_errors = []

    for true_domain, prompts in test_data.items():
        correct = 0
        rule_triggers = {}
        
        start_time = time.time()
        for prompt in prompts:
            # Pass verbose=False and log_file=None to keep terminal completely clean
            final_decision, logic_rule, detailed_log = router.predict(prompt, verbose=False, log_file=None)
            
            rule_triggers[logic_rule] = rule_triggers.get(logic_rule, 0) + 1
                
            if final_decision == true_domain:
                correct += 1
            else:
                # Store structured JSON data for the errors
                detailed_errors.append({
                    "true_domain": true_domain,
                    "predicted_domain": final_decision,
                    "rule_triggered": logic_rule,
                    "prompt": prompt,
                    "router_internal_log": detailed_log.strip()
                })
                
        elapsed = time.time() - start_time
        accuracy = (correct / len(prompts)) * 100
        
        total_correct += correct
        total_samples += len(prompts)
        
        # PURE TERMINAL OUTPUT
        print(f"[{true_domain.upper()}]")
        print(f"  -> Accuracy: {accuracy:>5.1f}%")
        print(f"  -> Speed:    {len(prompts)/elapsed:.1f} prompts/sec")
        print(f"  -> Handled by:")
        for rule, count in sorted(rule_triggers.items(), key=lambda x: x[1], reverse=True):
            print(f"       - {count:>3} : {rule}")
        print("-" * 70)

    # WRITE ONLY ERRORS TO JSON LOG FILE
    error_file_path = "logs/hybrid_router_errors.json"
    os.makedirs(os.path.dirname(error_file_path), exist_ok=True)
    with open(error_file_path, "w", encoding="utf-8") as f:
        json.dump(detailed_errors, f, indent=4, ensure_ascii=False)

    overall_accuracy = (total_correct / total_samples) * 100
    print(f"\nOVERALL HYBRID ACCURACY: {overall_accuracy:.2f}%")
    print(f"Detailed error logs saved to: {error_file_path}")
    print("="*70)

if __name__ == "__main__":
    main()