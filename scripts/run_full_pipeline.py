import os
import sys
import signal


# --- SSH/COLAB CTRL+C WORKAROUND ---
def ssh_interrupt_handler(signum, frame):
    raise KeyboardInterrupt()


try:
    # Maps Ctrl+\ (SIGQUIT) to act exactly like Ctrl+C in cloud environments
    signal.signal(signal.SIGQUIT, ssh_interrupt_handler)
except AttributeError:
    pass  # Ignore on Windows if testing locally
# -----------------------------

# --- WINDOWS LOCAL DEBUG FIX ---
# Must be set BEFORE importing torch to prevent OpenMP DLL conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import warnings
import argparse
from dotenv import load_dotenv
from huggingface_hub import login, hf_hub_download, snapshot_download

# Suppress Hugging Face warnings for clean debug logs
warnings.filterwarnings("ignore")

# Ensure the root directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.router.hybrid_router import HybridRouter
from src.model_ops.inference_manager import InferenceManager


def print_profile(step_name, start_time=None):
    """Prints elapsed time and current VRAM allocation."""
    elapsed = f"{(time.time() - start_time):.3f}s" if start_time else "N/A"
    allocated = (
        torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    )
    print(f"  [PROFILER] {step_name} | Time: {elapsed} | VRAM: {allocated:.2f} GB")


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Interactive Demo")

    # Unified repo containing both router files and adapter subfolders
    parser.add_argument(
        "--project_repo",
        type=str,
        default="ozgur-coban/confidence-driven-lora",
        help="Hugging Face repo containing all artifacts and subfolders",
    )
    parser.add_argument(
        "--use_few_shot_prompt",
        action="store_true",
        help="Enable 4-shot examples ",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load environment variables from .env file (for local runs)
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("  [AUTH] Hugging Face token found in environment. Logging in...")
        login(token=hf_token)
    else:
        print(
            "  [AUTH] Skipping environment HF_TOKEN (Assuming Colab userdata already handled login)."
        )

    print("=" * 70)
    print("HYBRID EXPERT LLM ROUTER - INTERACTIVE DEMO")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. BOOT SEQUENCE
    # ---------------------------------------------------------
    print("\n--- PHASE 1: INITIALIZATION ---")

    t0 = time.time()
    router = HybridRouter()

    # 1A. Fetch router artifacts from the 'router' subfolder
    print(f"  [INFO] Downloading router artifacts from {args.project_repo}...")
    try:
        semantic_path = hf_hub_download(
            repo_id=args.project_repo, filename="router/semantic_centroids.npy"
        )
        syntax_path = hf_hub_download(
            repo_id=args.project_repo, filename="router/syntax_router.joblib"
        )
    except Exception as e:
        print(f"\n  [ERROR] Failed to download router artifacts!")
        print(
            f"  Make sure '{args.project_repo}' contains the 'router/' subfolder with both files."
        )
        print(f"  Error details: {e}\n")
        sys.exit(1)

    # Load artifacts safely
    router.load(semantic_path=semantic_path, syntax_path=syntax_path)
    print_profile("Router Loaded", t0)

    t1 = time.time()

    # 1B. Fetch LoRA adapters from the 'adapters' subfolder
    print(f"  [INFO] Synchronizing LoRA adapters from {args.project_repo}...")
    try:
        # snapshot_download pulls the folders and returns the root cache path
        repo_cache_dir = snapshot_download(
            repo_id=args.project_repo, allow_patterns=["adapters/*"]
        )

        # Build exact absolute paths to pass to InferenceManager
        valid_adapters = {
            "code": os.path.join(repo_cache_dir, "adapters", "code"),
            "math": os.path.join(repo_cache_dir, "adapters", "math"),
            "medical": os.path.join(repo_cache_dir, "adapters", "medical"),
        }
    except Exception as e:
        print(f"\n  [ERROR] Failed to download LoRA adapters!")
        print(f"  Error details: {e}\n")
        sys.exit(1)

    # Initialize the inference manager (Base Model + All LoRAs)
    manager = InferenceManager(
        base_model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        adapter_paths=valid_adapters,
        use_few_shot_prompt=args.use_few_shot_prompt,
    )
    print_profile("Base Model & LoRAs Loaded", t1)

    # ---------------------------------------------------------
    # 2. INTERACTIVE EXECUTION LOOP
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("--- PHASE 2: INTERACTIVE EXECUTION ---")
    print("Type 'exit' or 'quit' to shut down.")
    print("Prefix your prompt with '/base ' to bypass the expert router.")
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
                prompt = prompt[
                    6:
                ].strip()  # Strip the "/base " command from the string
            elif prompt.lower() == "/base":
                print(
                    "  [ERROR] Please provide a prompt after the command. Example: /base What is 2+2?"
                )
                continue

            # --- ROUTING ---
            t_route = time.time()

            # Get the full distribution for debugging visibility
            sem_probs = router.semantic.predict_proba(prompt)
            syn_probs = router.syntax.predict_proba(prompt)

            domain, rule, detailed_log = router.predict(prompt, verbose=False)
            route_time = time.time() - t_route

            print("\n  [DEBUG: ROUTING PROBABILITIES]")
            print("    Semantic Distribution:")
            for k, v in sorted(
                sem_probs.items(), key=lambda item: item[1], reverse=True
            ):
                print(f"      - {k:<8}: {v:.4f}")

            print("    Syntax Distribution:")
            for k, v in sorted(
                syn_probs.items(), key=lambda item: item[1], reverse=True
            ):
                print(f"      - {k:<8}: {v:.4f}")

            print(
                f"\n  [ROUTING DECISION] Predicted Domain: {domain.upper()} | Rule: {rule} | Time: {route_time:.4f}s"
            )

            # --- GENERATION ---
            if bypass_lora:
                print(f"\n[LLAMA 3.1 - BASE MODEL ONLY (Routed as {domain.upper()})]:")
            else:
                print(f"\n[LLAMA 3.1 - {domain.upper()} EXPERT]:")
            print("-" * 50)

            t_gen = time.time()

            # --- THE INTERRUPT WRAPPER ---
            try:
                response = manager.generate(
                    user_prompt=prompt,
                    domain=domain,
                    stream=True,
                    bypass_lora=bypass_lora,
                )
            except KeyboardInterrupt:
                print("\n\n  [!] Generation interrupted by user.")
                # Force PyTorch to release the VRAM from the aborted generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("=" * 70)
                continue  # Instantly skip the metrics and loop back to the [USER]: prompt
            # -----------------------------

            gen_time = time.time() - t_gen
            print("-" * 50)

            word_count = len(response.split())
            speed = word_count / gen_time if gen_time > 0 else 0

            print_profile("Generation Complete", t_gen)
            print(
                f"  [DEBUG: METRICS] Generation Time: {gen_time:.2f}s | Approx Speed: {speed:.1f} words/sec"
            )
            print("=" * 70)

        except KeyboardInterrupt:
            # Exits cleanly if Ctrl+C is pressed at the empty [USER]: prompt
            print("\nShutting down pipeline...")
            break
        except Exception as e:
            print(f"\n[ERROR]: An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
