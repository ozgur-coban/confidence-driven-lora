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
        Applies chat template, tokenizes, truncates, and masks the prompt labels.
        """
        # 1. Apply the template to get the raw text string
        prompt_str = example["text"]

        # 2. Tokenize the string into integer IDs
        tokenized = self.tokenizer(
            prompt_str,
            truncation=True,
            max_length=self.max_length,
            padding=False, # Padding happens dynamically during batching
            add_special_tokens=True # Chat template already added the structural tokens
        )
        
        input_ids = tokenized["input_ids"]
        labels = input_ids.copy()
        
#3. Mask the labels using the string-split boundary method (highly robust)
        if self.response_template and self.response_template in prompt_str:
            # Reconstruct just the prompt portion
            prompt_only = prompt_str.split(self.response_template)[0] + self.response_template
            
            # Count exactly how many tokens make up the prompt
            prompt_token_len = len(self.tokenizer.encode(prompt_only, add_special_tokens=True))
            
            # Overwrite all prompt tokens with -100 so loss is only calculated on the answer
            labels[:prompt_token_len] = [-100] * prompt_token_len
        else:
            # Fallback: Mask everything if the template isn't found
            labels = [-100] * len(labels)            
        tokenized["labels"] = labels
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