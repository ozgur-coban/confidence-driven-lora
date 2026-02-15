import os, sys, ctypes

# Fix for Windows DLL loading (fixes WinError 127)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)
import argparse
import sys
from transformers import AutoTokenizer

# Ensure the src module can be found when running from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_ops.gsm8k_preprocessor import GSM8KPreprocessor
from src.model_ops.lora_manager import LoraManager
from src.training.trainer import AdapterTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train a single-domain LoRA adapter.")
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="NousResearch/Meta-Llama-3-8B-Instruct",
        help="Hugging Face model ID for the base model."
    )
    parser.add_argument(
        "--adapter_name", 
        type=str, 
        default="adapter_reasoning",
        help="Name of the output adapter directory."
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=512,
        help="Sequence truncation length."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"--- Starting Pipeline for {args.adapter_name} ---")

    # 1. Initialize Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # 2. Preprocess Data
    print("Preprocessing dataset...")
    preprocessor = GSM8KPreprocessor(tokenizer=tokenizer, max_length=args.max_length)
    processed_dataset = preprocessor.process()
    train_data = processed_dataset["train"]
    print(f"Training dataset ready: {len(train_data)} samples.")

    # 3. Setup Model Architecture
    print("Initializing model architecture...")
    lora_manager = LoraManager(model_id=args.model_id)
    base_model = lora_manager.load_base_model()
    peft_model = lora_manager.get_peft_setup(base_model)

    

    # 4. Execute Training
    trainer = AdapterTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        output_dir="./artifacts/adapters"
    )
    
    trainer.train(adapter_name=args.adapter_name)
    print("--- Pipeline Execution Complete ---")

if __name__ == "__main__":
    main()