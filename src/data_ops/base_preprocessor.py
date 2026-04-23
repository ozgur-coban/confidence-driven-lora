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
        """Maps raw dataset columns to the Llama 3 chat template dictionary format."""
        pass

    def tokenize_and_mask(self, example: dict) -> dict:
        """
        Applies chat template, tokenizes, truncates, and masks the prompt labels.
        """
        # 1. Apply the template to get the raw text string
        prompt_str = self.tokenizer.apply_chat_template(
            example["messages"], tokenize=False
        )

        # 2. Tokenize the string into integer IDs
        tokenized = self.tokenizer(
            prompt_str,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # Padding happens dynamically during batching
            add_special_tokens=False,  # Chat template already added the structural tokens
        )

        input_ids = tokenized["input_ids"]
        labels = input_ids.copy()

        # 3. Locate the boundary of the assistant's response
        # Llama 3 strictly uses this sequence to denote the assistant's turn
        response_template = "<|start_header_id|>assistant<|end_header_id|>"
        response_token_ids = self.tokenizer.encode(
            response_template, add_special_tokens=False
        )

        # Find where this exact sequence of tokens occurs in the input_ids array
        match_idx = -1
        for i in range(len(input_ids) - len(response_token_ids) + 1):
            if input_ids[i : i + len(response_token_ids)] == response_token_ids:
                # The actual answer we want to predict starts immediately AFTER this template
                match_idx = i + len(response_token_ids)
                break

        # 4. Mask the labels
        if match_idx != -1:
            # Overwrite all tokens before the assistant's answer with -100
            labels[:match_idx] = [-100] * match_idx
        else:
            # Fallback: If the template isn't found (e.g., massive truncation), mask everything.
            # This prevents the model from updating weights based on broken sequences.
            labels = [-100] * len(labels)

        tokenized["labels"] = labels
        return tokenized

    def process(self) -> DatasetDict:
        """Executes the data pipeline across the train and test splits."""
        if self.dataset is None:
            self.load_data()

        # Map the formatting logic and drop the raw string columns (question, answer)
        formatted_dataset = self.dataset.map(
            self.format_prompt, remove_columns=self.dataset["train"].column_names
        )

        # Map the tokenization and masking logic
        processed_dataset = formatted_dataset.map(self.tokenize_and_mask)

        # FILTER: Drop any sequence that reached the max_length limit to prevent training on truncated logic
        processed_dataset = processed_dataset.filter(
            lambda x: len(x["input_ids"]) < self.max_length
        )

        return processed_dataset
