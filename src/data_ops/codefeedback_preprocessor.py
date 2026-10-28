import os
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from .base_preprocessor import BasePreprocessor


class CodeFeedbackPreprocessor(BasePreprocessor):
    def __init__(
        self, tokenizer: AutoTokenizer, max_length: int = 512, data_size: int = 7500
    ) -> None:
        super().__init__(tokenizer, max_length)
        self.data_size = data_size

    def load_data(self) -> DatasetDict:
        """Loads CodeFeedback, filters natively for Python, downsamples, filters by length, and subsets."""
        print("Downloading m-a-p/CodeFeedback-Filtered-Instruction...")
        raw_dataset = load_dataset(
            "m-a-p/CodeFeedback-Filtered-Instruction", split="train"
        )

        # 1. Native Arrow filtering for Python (keeps memory footprint very low)
        print("Filtering strictly for Python samples...")
        python_dataset = raw_dataset.filter(
            lambda example: str(example.get("lang", "")).strip().lower() == "python",
            num_proc=4,
        )

        # 2. SPEED OPTIMIZATION: Slice a "safe pool" using native shuffle and select
        safe_pool_size = min(self.data_size * 10, len(python_dataset))
        print(f"Slicing to {safe_pool_size} Python samples before tokenization check...")
        pool_dataset = python_dataset.shuffle(seed=42).select(range(safe_pool_size))

        # 3. Placeholder for pure code extraction (currently using raw answer)
        print("Preparing text fields...")
        def prepare_fields(example):
            text = str(example.get("answer", ""))
            pattern = chr(96) * 3 + r"(?:python)?\n(.*?)" + chr(96) * 3
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                example["pure_code"] = match.group(1).strip()
            else:
                example["pure_code"] = text.strip()
            return example

        processed_pool = pool_dataset.map(prepare_fields, num_proc=4)
        processed_pool = processed_pool.filter(lambda x: len(x["pure_code"]) > 0, num_proc=4)

        # 4. FILTER LOGIC (Strict Token Length)
        def is_within_length(example):
            q = str(example.get("query", "")).strip()
            a = str(example.get("pure_code", "")).strip()
            full_text = f"Question: {q}\nAnswer:\n{a}{self.tokenizer.eos_token}"
            return len(self.tokenizer.encode(full_text, add_special_tokens=True)) <= self.max_length

        print(f"Filtering dataset for max_seq_length <= {self.max_length}...")
        filtered_dataset = processed_pool.filter(is_within_length, num_proc=4)
        print(
            f"Dropped {len(processed_pool) - len(filtered_dataset)} rows due to oversized tokens."
        )

        # 5. SUBSET LOGIC
        shuffled_dataset = filtered_dataset.shuffle(seed=42)
        subset_size = min(self.data_size, len(shuffled_dataset))
        subset_dataset = shuffled_dataset.select(range(subset_size))

        # 6. Exact 5% split for Trainer validation
        split_data = subset_dataset.train_test_split(test_size=0.05, seed=42)

        # 7. Standardize dictionary keys for train_adapter.py
        self.dataset = DatasetDict(
            {"train": split_data["train"], "validation": split_data["test"]}
        )

        return self.dataset

    def format_prompt(self, example: dict) -> dict:
        """Extracts the raw query and python code into standard keys."""
        return {
            # Map query to prompt, and pure_code to output for the base model formatter
            "prompt": str(example.get("query", "")).strip(),
            "output": str(example.get("pure_code", "")).strip(),
        }