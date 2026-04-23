import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from .base_preprocessor import BasePreprocessor
from config.domain_registry import DOMAIN_REGISTRY

class MedicalMeadowFlashcardsPreprocessor(BasePreprocessor):
    def __init__(self, tokenizer: AutoTokenizer, max_length: int=512, data_size: int=7500):
        super().__init__(tokenizer, max_length)
        self.data_size = data_size
        self.domain_cfg = DOMAIN_REGISTRY["medical"]

    def load_data(self) -> DatasetDict:
        """Loads, slices for speed, filters by strict length, and splits 5% for eval."""
        # 1. Load raw dataset
        raw_dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
        
        # 2. SPEED OPTIMIZATION: Slice a "safe pool" before the expensive filter
        safe_pool_size = min(self.data_size * 5, len(raw_dataset))
        print(f"Slicing to {safe_pool_size} samples before filtering for speed...")
        raw_dataset = raw_dataset.shuffle(seed=42).select(range(safe_pool_size))
        
        # 3. FILTER LOGIC
        def is_within_length(example):
            messages = self.format_prompt(example)["messages"]
            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            return len(self.tokenizer.encode(full_text)) <= self.max_length
            
        print(f"Filtering dataset for max_seq_length <= {self.max_length}")
        filtered_dataset = raw_dataset.filter(is_within_length, num_proc=4)
        print(f"Dropped {safe_pool_size - len(filtered_dataset)} oversized rows.")
        
        # 4. SUBSET LOGIC
        shuffled_dataset = filtered_dataset.shuffle(seed=42)
        subset_size = min(self.data_size, len(shuffled_dataset))
        subset_dataset = shuffled_dataset.select(range(subset_size))
        
        # 5.  Exact 5% split for Trainer validation
        split_data = subset_dataset.train_test_split(test_size=0.05, seed=42)
        
        # 6. Standardize dictionary keys for train_adapter.py
        self.dataset = DatasetDict({
            'train': split_data['train'],
            'validation': split_data['test']
        })
        
        return self.dataset

    def format_prompt(self, example: dict) -> dict:
        """
        Maps medical flashcards to Llama 3 format.
        CRITICAL: We use 'input' as the user content because 'instruction' is a static placeholder.
        """
        messages = [
            {
                "role": "system", 
                "content": self.domain_cfg["system"]
            },
            {
                "role": "user", 
                "content": str(example.get("input", "")).strip() # The REAL question is here
            },
            {
                "role": "assistant", 
                "content": str(example.get("output", "")).strip()
            }
        ]
        return {"messages": messages}