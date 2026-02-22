from datasets import load_dataset, DatasetDict
from .base_preprocessor import BasePreprocessor

class CodeAlpacaPreprocessor(BasePreprocessor):
    def load_data(self) -> DatasetDict:
        """Loads the CodeAlpaca dataset and creates a train/test split."""
        # CodeAlpaca only has a 'train' split by default
        raw_dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        
        # We hold out 5% (roughly 1,000 samples) for evaluation
        self.dataset = raw_dataset.train_test_split(test_size=0.05, seed=42)
        return self.dataset

    def format_prompt(self, example: dict) -> dict:
        """Maps the instruction, input, and output columns to the Llama 3 chat format."""
        
        # Handle the optional context/input field
        user_content = example["instruction"]
        if example.get("input") and str(example["input"]).strip() != "":
            user_content += f"\n\nContext/Input:\n{example['input']}"

        messages = [
            {
                "role": "system", 
                "content": "You are an expert programming assistant. Write clean, efficient, and well-documented code."
            },
            {
                "role": "user", 
                "content": user_content
            },
            {
                "role": "assistant", 
                "content": example["output"]
            }
        ]
        return {"messages": messages}