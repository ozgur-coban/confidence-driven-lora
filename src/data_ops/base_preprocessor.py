from abc import ABC, abstractmethod
from datasets import DatasetDict
from transformers import PreTrainedTokenizer

class BasePreprocessor(ABC):
    def __init__(self,response_template:str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.response_template=response_template
        self.dataset = None

    @abstractmethod
    def load_data(self) -> DatasetDict:
        """Loads the dataset from Hugging Face."""
        pass

    @abstractmethod
    def format_prompt(self, example: dict) -> dict:
        """Maps raw dataset columns to the Llama 3 chat template dictionary format."""
        pass

    def tokenize_and_mask(self, example: dict) -> dict:
        """
        Tokenizes the sequence. Label masking is now deferred to the TRL DataCollator.
        """
        prompt_str = example["text"]

        tokenized = self.tokenizer(
            prompt_str,
            truncation=True,
            max_length=self.max_length,
            padding=False, 
            add_special_tokens=True 
        )
        
        # We no longer create the "labels" array here.
        return tokenized

    def process(self) -> DatasetDict:
        """Executes the data pipeline across the train and test splits."""
        if self.dataset is None:
            self.dataset=self.load_data()
            
        # Map the formatting logic and drop the raw string columns (question, answer)
        formatted_dataset = self.dataset.map(
            self.format_prompt,
            remove_columns=self.dataset["train"].column_names
        )
        
        # Map the tokenization and masking logic
        processed_dataset = formatted_dataset.map(self.tokenize_and_mask)
        
        # FILTER: Drop any sequence that reached the max_length limit to prevent training on truncated logic
        processed_dataset = processed_dataset.filter(lambda x: len(x["input_ids"]) < self.max_length)
        
        return processed_dataset