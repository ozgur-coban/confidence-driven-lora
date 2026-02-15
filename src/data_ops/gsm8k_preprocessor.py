from datasets import load_dataset, DatasetDict
from .base_preprocessor import BasePreprocessor

class GSM8KPreprocessor(BasePreprocessor):
    def load_data(self) -> DatasetDict:
        """Loads the GSM8K dataset from Hugging Face."""
        # We specify "main" to grab the standard train/test splits
        self.dataset = load_dataset("openai/gsm8k", "main")
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