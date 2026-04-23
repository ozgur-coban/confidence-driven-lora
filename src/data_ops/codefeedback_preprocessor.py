import os
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from .base_preprocessor import BasePreprocessor
from config.domain_registry import DOMAIN_REGISTRY


class CodeFeedbackPreprocessor(BasePreprocessor):
    def __init__(
        self, tokenizer: AutoTokenizer, max_length: int = 512, data_size: int = 7500
    ):
        super().__init__(tokenizer, max_length)
        self.data_size = data_size
        self.domain_cfg = DOMAIN_REGISTRY["code"]

    def load_data(self) -> DatasetDict:
        """Loads CodeFeedback, filters for Python, slices for speed, extracts pure code, filters by length, and subsets."""
        print("Downloading m-a-p/CodeFeedback-Filtered-Instruction...")
        raw_dataset = load_dataset(
            "m-a-p/CodeFeedback-Filtered-Instruction", split="train"
        )
        df = raw_dataset.to_pandas()

        # 1. Filter strictly for Python (Fast Pandas operation)
        df["lang"] = df["lang"].astype(str).str.strip().str.lower()
        df = df[df["lang"] == "python"].copy()

        # 2. SPEED OPTIMIZATION: Slice a "safe pool" before expensive regex and tokenization
        safe_pool_size = min(self.data_size * 10, len(df))
        print(f"Slicing to {safe_pool_size} Python samples before regex extraction...")
        df = df.sample(n=safe_pool_size, random_state=42).copy()

        # 3. Extract pure code (dropping conversational fluff)
        # def extract_pure_code(text):
        #     pass

        print("Extracting pure code blocks...")
        # df["pure_code"] = df["answer"].apply(extract_pure_code)
        df["pure_code"] = df["answer"]

        # Drop rows where regex failed to find a standard code block
        df = df.dropna(subset=["pure_code"]).copy()

        # 4. Convert back to HuggingFace Dataset for fast tokenizer mapping
        hf_dataset = Dataset.from_pandas(df)

        # 5. FILTER LOGIC (Strict Token Length)
        def is_within_length(example):
            messages = self.format_prompt(example)["messages"]
            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            return len(self.tokenizer.encode(full_text)) <= self.max_length

        print(f"Filtering dataset for max_seq_length <= {self.max_length}...")
        filtered_dataset = hf_dataset.filter(is_within_length, num_proc=4)
        print(
            f"Dropped {len(hf_dataset) - len(filtered_dataset)} rows due to oversized tokens."
        )

        # 6. SUBSET LOGIC
        shuffled_dataset = filtered_dataset.shuffle(seed=42)
        subset_size = min(self.data_size, len(shuffled_dataset))
        subset_dataset = shuffled_dataset.select(range(subset_size))

        # 7. Exact 5% split for Trainer validation
        split_data = subset_dataset.train_test_split(test_size=0.05, seed=42)

        # 8. Standardize dictionary keys for train_adapter.py
        self.dataset = DatasetDict(
            {"train": split_data["train"], "validation": split_data["test"]}
        )

        return self.dataset

    def format_prompt(self, example: dict) -> dict:
        """
        Maps the pure algorithmic task to Llama 3 format.
        Uses the shared system prompt from the code domain registry.
        """
        messages = [
            {"role": "system", "content": self.domain_cfg["system"]},
            {"role": "user", "content": str(example.get("query", "")).strip()},
            {
                "role": "assistant",
                "content": str(
                    example.get("pure_code", "")
                ).strip(),  # Using the extracted code!
            },
        ]
        return {"messages": messages}
