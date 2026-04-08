import unsloth
import os, sys, ctypes

# Fix for Windows DLL loading (fixes WinError 127)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)
import argparse

# Ensure the src module can be found when running from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_ops.gsm8k_preprocessor import GSM8KPreprocessor
from src.data_ops.codealpaca_preprocessor import CodeAlpacaPreprocessor
from src.data_ops.medical_meadow_flashcards_preprocessor import MedicalMeadowFlashcardsPreprocessor
from src.data_ops.medqa_preprocessor import MedQAPreprocessor
from src.data_ops.metamath_preprocessor import MetaMathPreprocessor
from src.model_ops.lora_manager_unsloth import LoraManager
from src.training.trainer import AdapterTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train a single-domain LoRA adapter with Unsloth.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--adapter_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    
    # Training Arguments
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./artifacts/adapters")

    # LoRA Arguments
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--learning_rate", type=float, default=2e-4)

    parser.add_argument("--dataset", type=str, choices=["openai/gsm8k", "sahil2801/CodeAlpaca-20k","medalpaca/medical_meadow_medical_flashcards","GBaker/MedQA-USMLE-4-options","meta-math/MetaMathQA"], required=True)
    
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"--- Starting Unsloth Pipeline for {args.adapter_name} ---")

    # 1. Setup Model Architecture & Tokenizer First (Unsloth Flow)
    print("Initializing Unsloth model architecture and tokenizer...")
    lora_manager = LoraManager(model_id=args.model_id, max_length=args.max_length)
    base_model, tokenizer = lora_manager.load_base_model()
    
    peft_model = lora_manager.get_peft_setup(
        base_model, 
        r=args.r, 
        alpha=args.lora_alpha, 
        target_modules=args.target_modules
    )
    
    # 2. Preprocess Data (Now using the Unsloth tokenizer)
    print("Initializing Preprocessor...")
    if args.dataset == "openai/gsm8k":
        preprocessor = GSM8KPreprocessor(tokenizer, max_length=args.max_length)
    elif args.dataset == "sahil2801/CodeAlpaca-20k":
        preprocessor = CodeAlpacaPreprocessor(tokenizer, max_length=args.max_length)
    elif args.dataset == "medalpaca/medical_meadow_medical_flashcards":
        preprocessor = MedicalMeadowFlashcardsPreprocessor(tokenizer, max_length=args.max_length)
    elif args.dataset == "GBaker/MedQA-USMLE-4-options":
        preprocessor = MedQAPreprocessor(tokenizer, max_length=args.max_length)
    elif args.dataset == "meta-math/MetaMathQA":
        preprocessor= MetaMathPreprocessor(tokenizer, max_length=args.max_length)
    else:
        raise ValueError("Invalid dataset selected.")
        
    processed_dataset = preprocessor.process()
    train_data = processed_dataset["train"]
    print(f"Training dataset ready: {len(train_data)} samples.")

    # 3. Execute Training
    # Standard HF Trainer works flawlessly with Unsloth's patched model
    trainer = AdapterTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        output_dir=args.output_dir,
    )
    
    trainer.train(
        adapter_name=args.adapter_name,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate
    )
    print("--- Unsloth Pipeline Execution Complete ---")

if __name__ == "__main__":
    main()