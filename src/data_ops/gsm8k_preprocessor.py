from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from .base_preprocessor import BasePreprocessor

class GSM8KPreprocessor(BasePreprocessor):
    def __init__(self, tokenizer: AutoTokenizer, max_length: int=512, data_size: int=7500):
        super().__init__(tokenizer, max_length)
        self.data_size=data_size
        
    def load_data(self) -> DatasetDict:
        """Loads, slices for speed, filters by strict length, and splits 5% for eval."""
        # 1. Load raw dataset (Change the path based on the preprocessor)
        raw_dataset = load_dataset("openai/gsm8k", name="main", split="train")
        
        # 2. SPEED OPTIMIZATION: Slice a "safe pool" before the expensive filter
        #  grab 5x  need 
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
        """Maps the raw question and answer columns to the Llama 3 chat format."""
        messages = [
            {
                "role": "system", 
                "content": "You are a mathematical reasoning assistant. Solve the problem step-by-step and provide the final answer after ####."
            },
            {
                "role": "user", 
                "content": example["question"]
            },
            {
                "role": "assistant", 
                "content": example["answer"]
            }
        ]
        return {"messages": messages}