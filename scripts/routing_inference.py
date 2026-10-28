import os
import warnings

# --- WINDOWS LOCAL DEBUG FIX ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Suppress Hugging Face warnings for clean debug logs
warnings.filterwarnings("ignore")
import sys
import time
import torch
import threading

# --- CROSS-PLATFORM NON-BLOCKING INPUT ---
if os.name == "nt":
    import msvcrt
else:
    import select
# -----------------------------------------

import argparse
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import StoppingCriteria, StoppingCriteriaList

# Ensure the root directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.router.meta_router import MetaRouter
from src.model_ops.inference_manager import InferenceManager


# --- PYTORCH NATIVE INTERRUPT HANDLER ---
class InterruptStoppingCriteria(StoppingCriteria):
    def __init__(self):
        self.is_interrupted = False

    def __call__(self, input_ids, scores, **kwargs):
        # The generation loop checks this after every token.
        return self.is_interrupted


global_interrupt = InterruptStoppingCriteria()
# Notice: All signal hooks have been completely removed.
# ---------------------------------------------


def print_profile(step_name, start_time=None):
    """Prints elapsed time and current VRAM allocation."""
    elapsed = f"{(time.time() - start_time):.3f}s" if start_time else "N/A"
    allocated = (
        torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    )
    print(f"  [PROFILER] {step_name} | Time: {elapsed} | VRAM: {allocated:.2f} GB")


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Debug Test")
    parser.add_argument(
        "--code_adapter",
        type=str,
        default="./models/adapters/adapter_code/code_codefeedback_3b_base_07.06.2026",
    )
    parser.add_argument(
        "--math_adapter",
        type=str,
        default="./models/adapters/adapter_math/math_gsm8k_main_3b_base_07.06.2026",
    )
    parser.add_argument(
        "--hybrid_adapter",
        type=str,
        default="./models/adapters/adapter_hybrid/math_code_ties_07.06.2026/hybrid"
    )
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
        print(
            "  [WARNING] HF_TOKEN not found in .env file. Gated models will fail to load."
        )
    # ---------------------------------------
    print("=" * 70)
    print("END-TO-END PIPELINE DEBUG TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. BOOT SEQUENCE
    # ---------------------------------------------------------
    print("\n--- PHASE 1: INITIALIZATION ---")

    t0 = time.time()
    router = MetaRouter()

    # Ensure all 3 router artifacts exist before trying to load them
    semantic_path = "models/semantic_centroids.npy"
    syntax_path = "models/syntax_router.joblib"
    meta_path = "models/meta_router.joblib"

    if not (os.path.exists(semantic_path) and os.path.exists(syntax_path) and os.path.exists(meta_path)):
        print("\n  [ERROR] Router artifacts not found!")
        print("  Please build them manually")
        sys.exit(1)

    # Load artifacts safely
    router.load(
        semantic_path=semantic_path, 
        syntax_path=syntax_path, 
        meta_path=meta_path
    )

    print_profile("Router Loaded", t0)

    t1 = time.time()
    raw_adapter_paths = {
        "code": args.code_adapter,
        "math": args.math_adapter,
        "hybrid": args.hybrid_adapter,
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
        base_model_id="meta-llama/Llama-3.2-3B",
        adapter_paths=valid_adapters,
    )
    print_profile("Base Model & LoRAs Loaded", t1)

    # ---------------------------------------------------------
    # 2. INTERACTIVE EXECUTION LOOP
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("--- PHASE 2: INTERACTIVE EXECUTION ---")
    print("Type 'exit' or 'quit' to shut down.")
    print("Press ENTER at any time during generation to instantly halt it.")
    print("=" * 70)

    while True:
        try:
            raw_prompt = input("\n[USER]: ")
            if raw_prompt.lower() in ["exit", "quit"]:
                print("\nShutting down pipeline...")
                break
            if not raw_prompt.strip():
                continue

            # --- PARAMETER INTERCEPTION ---
            bypass_lora = False
            prompt = raw_prompt.strip()

            if prompt.lower().startswith("/base "):
                bypass_lora = True
                prompt = prompt[6:].strip()  # Strip the "/base " command from the string
            elif prompt.lower() == "/base":
                print(
                    "  [ERROR] Please provide a prompt after the command. Example: /base What is 2+2?"
                )
                continue

            # --- ROUTING ---
            t_route = time.time()
            
            # Fetch raw probabilities from sub-routers first
            sem_probs = router.semantic.predict_proba(prompt)
            syn_probs = router.syntax.predict_proba(prompt)

            # Predict utilizing the full 6-dimensional Meta-Classifier
            domain, rule, detailed_log = router.predict(prompt, verbose=False, log_file=None)
            route_time = time.time() - t_route

            print("\n  [DEBUG: FULL CLASS PROBABILITIES]")
            print("    Semantic Distribution:")
            for k, v in sorted(sem_probs.items(), key=lambda item: item[1], reverse=True):
                print(f"      - {k:<8}: {v:.4f}")

            print("    Syntax Distribution:")
            for k, v in sorted(syn_probs.items(), key=lambda item: item[1], reverse=True):
                print(f"      - {k:<8}: {v:.4f}")

            print(f"\n  [DEBUG: ROUTER INTERNAL LOG]\n{detailed_log}")
            print(
                f"\n  [DEBUG: ROUTING DECISION] Predicted Domain: {domain.upper()} | Rule: {rule} | Time: {route_time:.3f}s"
            )
            print_profile(f"Routing Complete ({domain.upper()})", t_route)

            # --- GENERATION ---
            if bypass_lora:
                print(f"\n[LLAMA 3.2 (3B) - BASE MODEL ONLY (Routed as {domain.upper()})]:")
            else:
                print(f"\n[LLAMA 3.2 (3B) - {domain.upper()} EXPERT]:")
            print("-" * 50)

            t_gen = time.time()

            # Reset the interrupt flag before starting generation
            global_interrupt.is_interrupted = False
            response_container = [""]

            def generation_task():
                response_container[0] = manager.generate(
                    prompt,
                    domain,
                    stream=True,
                    bypass_lora=bypass_lora,
                    stopping_criteria=StoppingCriteriaList([global_interrupt]),
                )

            gen_thread = threading.Thread(target=generation_task)
            gen_thread.start()

            # --- NON-BLOCKING INPUT LISTENER (BYPASSES SSH CTRL+C ISSUES) ---
            interrupted_by_user = False

            # Flush the input buffer before listening so accidental earlier keystrokes don't trigger it
            if os.name == "nt":
                while msvcrt.kbhit():
                    msvcrt.getch()
            else:
                while select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.read(1)

            # Main thread loops while model generates
            while gen_thread.is_alive():
                gen_thread.join(0.1)  # Wait a tiny bit

                if os.name == "nt":
                    if msvcrt.kbhit():
                        msvcrt.getch()  # Consume the keypress
                        interrupted_by_user = True
                        break
                else:
                    # Unix/Linux (SSH environments)
                    i, _, _ = select.select([sys.stdin], [], [], 0)
                    if i:
                        sys.stdin.readline()  # Consume the enter keypress
                        interrupted_by_user = True
                        break

            if interrupted_by_user:
                print(
                    "\n\n  [!] Generation forcefully halted by user (ENTER key pressed)."
                )
                global_interrupt.is_interrupted = True
                gen_thread.join()  # Wait for the thread to exit cleanly

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("=" * 70)
                continue  # Loop back to [USER]: prompt instantly
            # -----------------------------

            response = response_container[0]
            gen_time = time.time() - t_gen

            print("-" * 50)

            word_count = len(response.split())
            speed = word_count / gen_time if gen_time > 0 else 0

            print_profile("Generation Complete", t_gen)
            print(
                f"  [DEBUG: METRICS] Generation Time: {gen_time:.2f}s | Approx Speed: {speed:.1f} words/sec"
            )
            print("=" * 70)

        # Catch Ctrl+C ONLY when the user is typing at the empty [USER]: prompt
        except KeyboardInterrupt:
            print("\nShutting down pipeline...")
            break
        except Exception as e:
            print(f"\n[ERROR]: An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()