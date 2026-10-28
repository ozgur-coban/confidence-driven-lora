import os
import argparse
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Offline TIES Merge for Math and Code LoRAs.")
    parser.add_argument(
        "--base_model_id", type=str, default="meta-llama/Llama-3.2-3B"
    )
    parser.add_argument(
        "--math_adapter", 
        type=str, 
        default="./models/adapters/adapter_math/math_gsm8k_main_3b_base_07.06.2026",
        help="Path to the trained math adapter."
    )
    parser.add_argument(
        "--code_adapter", 
        type=str, 
        default="./models/adapters/adapter_code/code_codefeedback_3b_base_07.06.2026",
        help="Path to the trained code adapter."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./models/adapters/adapter_hybrid/math_code_ties_07.06.2026",
        help="Where to save the newly fused hybrid adapter."
    )
    parser.add_argument(
        "--density", 
        type=float, 
        default=0.8, 
        help="TIES density parameter (0.0 to 1.0). Retains top-K magnitudes."
    )
    parser.add_argument(
        "--weight", 
        type=float, 
        default=0.5, 
        help="Equal scaling weight for both adapters (0.5 + 0.5 = 1.0)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n=== STARTING OFFLINE TIES MERGE ===")
    
    # 1. Load Base Model into CPU memory 
    # (FP16 reduces RAM usage while maintaining LoRA compatibility precision)
    print(f"1. Loading base model ({args.base_model_id})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, 
        torch_dtype=torch.float16,
        device_map="cpu" 
    )

    # 2. Mount the Math Adapter
    print(f"2. Mounting MATH adapter from: {args.math_adapter}")
    if not os.path.exists(args.math_adapter):
        raise FileNotFoundError(f"Math adapter not found at {args.math_adapter}")
    model = PeftModel.from_pretrained(base_model, args.math_adapter, adapter_name="math")

    # 3. Mount the Code Adapter
    print(f"3. Mounting CODE adapter from: {args.code_adapter}")
    if not os.path.exists(args.code_adapter):
        raise FileNotFoundError(f"Code adapter not found at {args.code_adapter}")
    model.load_adapter(args.code_adapter, adapter_name="code")

    # 4. Execute the TIES fusion
    print(f"4. Executing TIES Merge (Density: {args.density}, Weights: {args.weight} / {args.weight})...")
    model.add_weighted_adapter(
        adapters=["math", "code"],
        weights=[args.weight, args.weight],
        adapter_name="hybrid",
        combination_type="ties",
        density=args.density
    )

    # 5. Extract and Save Only the Hybrid Target
    print(f"5. Saving fused HYBRID adapter to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Passing the specific adapter_name ensures we don't save the base weights or the unmerged LoRAs
    model.save_pretrained(args.output_dir, adapter_name="hybrid")
    
    print("=== TIES MERGE COMPLETE ===")
    print(f"Update your InferenceManager adapter_paths with:\n'hybrid': '{args.output_dir}'")


if __name__ == "__main__":
    main()