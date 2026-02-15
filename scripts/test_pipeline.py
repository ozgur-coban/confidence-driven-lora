from transformers import AutoTokenizer
import sys
import os

# Ensure the src module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_ops.gsm8k_preprocessor import GSM8KPreprocessor

def test_preprocessing():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")
    
    print("Initializing Preprocessor...")
    # Setting max_length to 512 based on our EDA results
    preprocessor = GSM8KPreprocessor(tokenizer=tokenizer, max_length=512)
    
    print("Running pipeline (Downloading, Formatting, Tokenizing, Masking)...")
    processed_data = preprocessor.process()
    
    print("\n--- Verification ---")
    sample = processed_data["train"][0]
    
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    
    print(f"Total tokens in sequence: {len(input_ids)}")
    
    # Count how many tokens are masked vs unmasked
    masked_count = labels.count(-100)
    unmasked_count = len(labels) - masked_count
    print(f"Tokens masked (-100, ignored by loss): {masked_count}")
    print(f"Tokens unmasked (actively trained on): {unmasked_count}")

if __name__ == "__main__":
    test_preprocessing()