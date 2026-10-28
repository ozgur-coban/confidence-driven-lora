from abc import ABC, abstractmethod
from datasets import DatasetDict
from transformers import PreTrainedTokenizer


class BasePreprocessor(ABC):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = None

    @abstractmethod
    def load_data(self) -> DatasetDict:
        """Loads the dataset from Hugging Face."""
        pass

    @abstractmethod
    def format_prompt(self, example: dict) -> dict:
        """
        Maps raw dataset columns to a standard string dictionary format:
        {"prompt": "...", "output": "..."}
        """
        pass

    def process(self) -> DatasetDict:
        """Executes the data pipeline across the train and test splits."""
        if self.dataset is None:
            self.load_data()
        available_splits=list(self.dataset.keys())
        first_split=available_splits[0]
        original_columns=self.dataset[first_split].column_names

        # Map the formatting logic to create standard 'prompt' and 'output' columns,
        # and drop all original raw string columns from the Hugging Face dataset.
        processed_dataset = self.dataset.map(
            self.format_prompt, 
            remove_columns=original_columns,
            num_proc=4
        )

        return processed_dataset