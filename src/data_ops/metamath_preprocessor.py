from datasets import load_dataset, DatasetDict
from .base_preprocessor import BasePreprocessor


class MetaMathPreprocessor(BasePreprocessor):
    def __init__(self, tokenizer, max_length=512, data_size=15000):
        super().__init__(tokenizer, max_length)
        self.data_size = data_size

    def load_data(self) -> DatasetDict:
        """Loads, slices for speed, filters by strict length, and splits 5% for eval."""
        # 1. Load raw dataset (Change the path based on the preprocessor)
        raw_dataset = load_dataset("meta-math/MetaMathQA", split="train")

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
        self.dataset = DatasetDict(
            {"train": split_data["train"], "validation": split_data["test"]}
        )

        return self.dataset

    def _standardize_target(self, response_text: str) -> str:
        """
        Normalizes MetaMathQA responses to strictly end with '#### [answer]'.
        Extracts the explicit answer provided after 'The answer is: '.
        """
        text = str(response_text)
        if "The answer is:" in text:
            parts = text.split("The answer is:")
            main_logic = parts[0].strip()
            final_answer = parts[1].strip()

            if "####" in main_logic:
                main_logic = main_logic.split("####")[0].strip()

            return f"{main_logic}\n#### {final_answer}"

        return text

    def format_prompt(self, example: dict) -> dict:
        """Maps the raw columns to the Llama 3 chat format."""
        standardized_ans = self._standardize_target(example["response"])

        messages = [
            {
                "role": "system",
                "content": "You are a mathematical reasoning assistant. Solve the problem step-by-step and provide the final answer after ####.",
            },
            {"role": "user", "content": example["query"]},
            {"role": "assistant", "content": standardized_ans},
        ]
        return {"messages": messages}
