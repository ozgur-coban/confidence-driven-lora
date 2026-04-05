import os
import sys

# --- WINDOWS LOCAL DEBUG FIX ---
# Must be set BEFORE importing torch to prevent OpenMP DLL conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import warnings
import argparse
import subprocess
from dotenv import load_dotenv
from huggingface_hub import login

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
    parser.add_argument("--code_adapter", type=str, default="./models/adapters/adapter_codefeedback_algo/final")
    parser.add_argument("--math_adapter", type=str, default="./models/adapters/adapter_reasoning/final")
    parser.add_argument("--medical_adapter", type=str, default="./models/adapters/adapter_medqa/final")
    return parser.parse_args()

def main():
    args = parse_args()
    # Load environment variables from .env file
    load_dotenv()
    
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("  [AUTH] Hugging Face token found. Logging in...")
        login(token=hf_token)
    else:
        print("  [WARNING] HF_TOKEN not found in .env file. Gated models will fail to load.")
    # ---------------------------------------
    print("="*70)
    print("END-TO-END PIPELINE DEBUG TEST")
    print("="*70)

    # ---------------------------------------------------------
    # 1. BOOT SEQUENCE
    # ---------------------------------------------------------
    print("\n--- PHASE 1: INITIALIZATION ---")
    
    t0 = time.time()
    router = HybridRouter()
    
    # Ensure router artifacts exist before trying to load them
    semantic_path = "models/semantic_centroids.npy"
    syntax_path = "models/syntax_router.joblib"
    
    if not (os.path.exists(semantic_path) and os.path.exists(syntax_path)):
        print("\n  [ERROR] Router artifacts not found!")
        print("  Please build them manually by running:")
        print("    python -m scripts.build_semantic_router --enable_training")
        print("    python -m scripts.build_syntax_router --enable_training\n")
        sys.exit(1)

    # Load artifacts safely
    router.load(semantic_path=semantic_path, syntax_path=syntax_path)
        
    print_profile("Router Loaded", t0)

    t1 = time.time()
    raw_adapter_paths = {
        "code": args.code_adapter,
        "math": args.math_adapter,
        "medical": args.medical_adapter
    }
    
    # Check which adapters actually exist to prevent PEFT crashes
    valid_adapters = {}
    for domain, path in raw_adapter_paths.items():
        if os.path.exists(path):
            valid_adapters[domain] = path
        else:
            print(f"  [WARNING] Missing adapter for '{domain}' at: {path}")

    if not valid_adapters:
        print("  [WARNING] No valid adapters found. Running in BASE MODEL ONLY mode.")

    manager = InferenceManager(
        base_model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        adapter_paths=valid_adapters
    )
    print_profile("Base Model & LoRAs Loaded", t1)

    # ---------------------------------------------------------
    # 2. INTERACTIVE EXECUTION LOOP
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print("--- PHASE 2: INTERACTIVE EXECUTION ---")
    print("Type 'exit' or 'quit' to shut down.")
    print("="*70)
    
    while True:
        try:
            raw_prompt = input("\n[USER]: ")
            if raw_prompt.lower() in ['exit', 'quit']:
                print("\nShutting down pipeline...")
                break
            if not raw_prompt.strip():
                continue
                
            # --- PARAMETER INTERCEPTION ---
            bypass_lora = False
            prompt = raw_prompt.strip()
            
            if prompt.lower().startswith("/base "):
                bypass_lora = True
                prompt = prompt[6:].strip() # Strip the "/base " command from the string
            elif prompt.lower() == "/base":
                print("  [ERROR] Please provide a prompt after the command. Example: /base What is 2+2?")
                continue

            # --- ROUTING ---
            t_route = time.time()
            
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
            if bypass_lora:
                print(f"\n[LLAMA 3.1 - BASE MODEL ONLY (Routed as {domain.upper()})]:")
            else:
                print(f"\n[LLAMA 3.1 - {domain.upper()} EXPERT]:")
            print("-" * 50)
            
            t_gen = time.time()
            
            # --- THE INTERRUPT WRAPPER ---
            try:
                # Ensure inference_manager.py's generate function accepts bypass_lora
                response = manager.generate(prompt, domain, stream=True, bypass_lora=bypass_lora)
            except KeyboardInterrupt:
                print("\n\n  [!] Generation interrupted by user.")
                # Force PyTorch to release the VRAM from the aborted generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("="*70)
                continue # Instantly skip the metrics and loop back to the [USER]: prompt
            # -----------------------------
            
            gen_time = time.time() - t_gen
            
            print("-" * 50)
            
            word_count = len(response.split())
            speed = word_count / gen_time if gen_time > 0 else 0

            print_profile("Generation Complete", t_gen)
            print(f"  [DEBUG: METRICS] Generation Time: {gen_time:.2f}s | Approx Speed: {speed:.1f} words/sec")
            print("="*70)

        except KeyboardInterrupt:
            # If you press Ctrl+C while sitting at the empty [USER]: prompt, it exits the script
            print("\nShutting down pipeline...")
            break
        except Exception as e:
            print(f"\n[ERROR]: An unexpected error occurred: {e}")
if __name__ == "__main__":
    main()