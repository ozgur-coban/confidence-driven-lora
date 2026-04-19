import os, sys, ctypes

# Fix for Windows DLL loading (fixes WinError 127)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)
import argparse
from transformers import AutoTokenizer
from datasets import concatenate_datasets # Added for mixed datasets

# Ensure the src module can be found when running from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_ops.gsm8k_preprocessor import GSM8KPreprocessor
from src.data_ops.codealpaca_preprocessor import CodeAlpacaPreprocessor
from src.data_ops.medical_meadow_flashcards_preprocessor import MedicalMeadowFlashcardsPreprocessor
from src.data_ops.medqa_preprocessor import MedQAPreprocessor
from src.data_ops.metamath_preprocessor import MetaMathPreprocessor
from src.model_ops.lora_manager import LoraManager
from src.training.trainer import AdapterTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train a single-domain or mixed LoRA adapter.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--data_size", type=int, default=7500, help="Number of samples PER DOMAIN.")
    
    # CLI ARGUMENTS
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./models/adapters", help="Base directory to save the trained adapters.")
    parser.add_argument("--save_steps", type=int, default=100, help="How often to save and evaluate.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a specific checkpoint folder to resume training.")

    # LORA arguments
    parser.add_argument("--r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], help="List of target modules for LoRA.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate.")

    parser.add_argument("--dataset", type=str, choices=[
        "openai/gsm8k", 
        "sahil2801/CodeAlpaca-20k",
        "medalpaca/medical_meadow_medical_flashcards",
        "GBaker/MedQA-USMLE-4-options",
        "meta-math/MetaMathQA",
        "mixed" #Triggers the combined dataset logic
    ], required=True, help="Which dataset/domain to train on.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"--- Starting Pipeline for {args.adapter_name} ---")

    # 1. Initialize Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # 2. Preprocess Data
    print("Initializing Preprocessor...")
    
    if args.dataset == "mixed":
        print(f"--- Building Mixed Dataset ({args.data_size} samples per domain) ---")
        # Initialize the three core preprocessors
        p_math = GSM8KPreprocessor(tokenizer, max_length=args.max_length, data_size=args.data_size)
        p_code = CodeAlpacaPreprocessor(tokenizer, max_length=args.max_length, data_size=args.data_size)
        p_med = MedicalMeadowFlashcardsPreprocessor(tokenizer, max_length=args.max_length, data_size=args.data_size)
        
        # Process all three
        print("Processing Math Data...")
        ds_math = p_math.process()
        print("Processing Code Data...")
        ds_code = p_code.process()
        print("Processing Medical Data...")
        ds_med = p_med.process()
        
        # Combine and Shuffle
        print("Concatenating and shuffling datasets...")
        train_data = concatenate_datasets([ds_math["train"], ds_code["train"], ds_med["train"]]).shuffle(seed=42)
        eval_data = concatenate_datasets([ds_math["validation"], ds_code["validation"], ds_med["validation"]]).shuffle(seed=42)
        
    else:
        # Standard Single-Dataset Logic
        if args.dataset == "openai/gsm8k":
            preprocessor = GSM8KPreprocessor(tokenizer, max_length=args.max_length, data_size=args.data_size)
        elif args.dataset == "sahil2801/CodeAlpaca-20k":
            preprocessor = CodeAlpacaPreprocessor(tokenizer, max_length=args.max_length, data_size=args.data_size)
        elif args.dataset == "medalpaca/medical_meadow_medical_flashcards":
            preprocessor = MedicalMeadowFlashcardsPreprocessor(tokenizer, max_length=args.max_length, data_size=args.data_size)
        elif args.dataset == "GBaker/MedQA-USMLE-4-options":
            preprocessor = MedQAPreprocessor(tokenizer, max_length=args.max_length, data_size=args.data_size)
        elif args.dataset == "meta-math/MetaMathQA":
            preprocessor = MetaMathPreprocessor(tokenizer, max_length=args.max_length, data_size=args.data_size)
        else:
            raise ValueError("Invalid dataset selected.")
            
        processed_dataset = preprocessor.process()
        train_data = processed_dataset["train"]
        if("validation" in processed_dataset):
            eval_data = processed_dataset["validation"]
        else:
            eval_data = processed_dataset["test"]

    print(f"Training dataset ready: {len(train_data)} samples.")
    print(f"Evaluation dataset ready: {len(eval_data)} samples.")

    # 3. Setup Model Architecture
    print("Initializing model architecture...")
    lora_manager = LoraManager(model_id=args.model_id)
    base_model = lora_manager.load_base_model()
    peft_model = lora_manager.get_peft_setup(
        base_model, 
        r=args.r, 
        alpha=args.lora_alpha, 
        target_modules=args.target_modules
    )

    # 4. Final Verification & Execute Training
    print(f"\n{'='*60}")
    print("  TRAINING PARAMS")
    print(f"{'='*60}")
    print(f" Target Adapter:    {args.adapter_name}")
    print(f" Total Train Rows:  {len(train_data)}")
    print(f" Total Eval Rows:   {len(eval_data)}")
    print(f" Batch Size:        {args.batch_size}")
    print(f" Gradient Accum:    {args.grad_accum}")
    print(f" Learning Rate:     {args.learning_rate}")
    print(f"{'='*60}\n")
    trainer = AdapterTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        output_dir=args.output_dir,
    )
    
    trainer.train(
        adapter_name=args.adapter_name,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    print("--- Pipeline Execution Complete ---")

if __name__ == "__main__":
    main()