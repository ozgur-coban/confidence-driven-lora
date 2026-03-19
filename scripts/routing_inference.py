import os
import sys
import time
import torch
import warnings
import argparse
import subprocess

# Suppress Hugging Face warnings for clean debug logs
warnings.filterwarnings("ignore")

# Ensure the root directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.router.hybrid_router import HybridRouter
from src.model_ops.inference_manager import InferenceManager

def print_profile(step_name, start_time=None):
    """Prints elapsed time and current VRAM allocation."""
    elapsed = f"{(time.time() - start_time):.3f}s" if start_time else "N/A"
    allocated = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    print(f"  [PROFILER] {step_name} | Time: {elapsed} | VRAM: {allocated:.2f} GB")

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Debug Test")
    parser.add_argument("--code_adapter", type=str, default="/content/drive/MyDrive/capstone-project/training_results/adapters/adapter_codefeedback_algo/final")
    parser.add_argument("--math_adapter", type=str, default="/content/drive/MyDrive/capstone-project/training_results/adapters/adapter_gsm8k/final")
    parser.add_argument("--medical_adapter", type=str, default="/content/drive/MyDrive/capstone-project/training_results/adapters/adapter_medqa/final")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("="*70)
    print("END-TO-END PIPELINE DEBUG TEST")
    print("="*70)

    # ---------------------------------------------------------
    # 1. BOOT SEQUENCE
    # ---------------------------------------------------------
    print("\n--- PHASE 1: INITIALIZATION ---")
    
    t0 = time.time()
    router = HybridRouter()
    
    # Attempt to load router artifacts; fallback to training if missing
    try:
        router.load(
            semantic_path="models/semantic_centroids.npy", 
            syntax_path="models/syntax_router.joblib"
        )
    except FileNotFoundError:
        print("  [PROFILER] Router artifacts not found. Initiating training scripts...")
        # Assuming your training scripts are located in the scripts folder
        subprocess.run([sys.executable, "scripts/build_semantic_router.py"], check=True)
        subprocess.run([sys.executable, "scripts/build_syntax_router.py"], check=True)
        # Reload after successful build
        router.load(
            semantic_path="models/semantic_centroids.npy", 
            syntax_path="models/syntax_router.joblib"
        )
        
    print_profile("Router Loaded", t0)

    t1 = time.time()
    adapter_paths = {
        "code": args.code_adapter,
        "math": args.math_adapter,
        "medical": args.medical_adapter
    }
    
    manager = InferenceManager(
        base_model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        adapter_paths=adapter_paths
    )
    print_profile("Base Model & All LoRAs Loaded", t1)

    # ---------------------------------------------------------
    # 2. TEST QUERIES
    # ---------------------------------------------------------
    test_queries = {
        "Expected Medical": "A 45-year-old male presents with sudden onset of severe, tearing chest pain radiating to his back. His blood pressure is 180/110 mmHg. A chest X-ray shows a widened mediastinum. What is the most likely diagnosis and immediate next step in management?",
        "Expected Math": "If I have a rectangular garden that is 12 feet long and 8 feet wide, and I want to put a 2-foot wide gravel path around the entire outside, what is the total area of the gravel path? ####"
    }

    print("\n" + "="*70)
    print("--- PHASE 2: EXECUTION ---")
    
    for label, prompt in test_queries.items():
        print(f"\n[TEST CASE]: {label}")
        print(f"[PROMPT]: {prompt[:100]}...")
        
        # --- ROUTING ---
        t_route = time.time()
        
        # Pull raw probability distributions directly from the sub-routers for deep debugging
        sem_probs = router.semantic.predict_proba(prompt)
        syn_probs = router.syntax.predict_proba(prompt)
        
        domain, rule, detailed_log = router.predict(prompt, verbose=False)
        route_time = time.time() - t_route
        
        print("\n  [DEBUG: FULL CLASS PROBABILITIES]")
        print("    Semantic Distribution:")
        for k, v in sorted(sem_probs.items(), key=lambda item: item[1], reverse=True):
            print(f"      - {k:<8}: {v:.4f}")
            
        print("    Syntax Distribution:")
        for k, v in sorted(syn_probs.items(), key=lambda item: item[1], reverse=True):
            print(f"      - {k:<8}: {v:.4f}")
        
        print(f"\n  [DEBUG: ROUTING DECISION] Predicted Domain: {domain.upper()} | Rule: {rule} | Time: {route_time:.3f}s")
        print_profile(f"Routing Complete ({domain.upper()})", t_route)

        # --- GENERATION ---
        t_gen = time.time()
        response = manager.generate(prompt, domain)
        gen_time = time.time() - t_gen
        
        # Calculate approximate generation speed (words/sec is a rough proxy for tokens/sec here)
        word_count = len(response.split())
        speed = word_count / gen_time if gen_time > 0 else 0

        print_profile("Generation Complete", t_gen)
        
        print("\n  [DEBUG: FINAL OUTPUT]")
        print("-" * 50)
        print(response)
        print("-" * 50)
        print(f"  [DEBUG: METRICS] Generation Time: {gen_time:.2f}s | Approx Speed: {speed:.1f} words/sec")
        print("="*70)

if __name__ == "__main__":
    main()