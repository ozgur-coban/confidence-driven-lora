from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from .base_preprocessor import BasePreprocessor

class GSM8KPreprocessor(BasePreprocessor):
    def __init__(self, tokenizer: AutoTokenizer, max_length: int=512, data_size: int=7500):
        super().__init__(tokenizer, max_length)
        self.data_size=data_size
        
    def load_data(self) -> DatasetDict:
        """Loads the GSM8K dataset from Hugging Face."""
        self.dataset = load_dataset("openai/gsm8k", "main", split="train")
        def is_within_length(example):
            messages=self.format_prompt(example)
            full_text=self.tokenizer.apply_chat_template(messages,tokenize=False)
            return len(self.tokenizer.encode(full_text))<=self.max_length
        filtered_dataset=self.dataset.filter(is_within_length,num_proc=4)
        print(f"Dropped {len(self.dataset) - len(filtered_dataset)} oversized rows.")
        shuffled_dataset=filtered_dataset.shuffle(seed=42)
        subset_size=min(self.data_size,len(shuffled_dataset))
        subset_dataset=shuffled_dataset.select(range(subset_size))
        self.dataset=subset_dataset.train_test_split(test_size=0.05,seed=42)
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