import re
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from .base_preprocessor import BasePreprocessor

class CodeFeedbackPreprocessor(BasePreprocessor):
    def load_data(self) -> DatasetDict:
        """Loads CodeFeedback, filters for Python, extracts pure code, filters by length, and subsets."""
        print("Downloading m-a-p/CodeFeedback-Filtered-Instruction...")
        raw_dataset = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
        df = raw_dataset.to_pandas()
        
        # 1. Filter strictly for Python
        df["lang"] = df["lang"].astype(str).str.strip().str.lower()
        df = df[df["lang"] == "python"].copy()
        
        # 2. Extract pure code (dropping conversational fluff)
        def extract_pure_code(text):
            match = re.search(r'```(?:python)?(.*?)```', str(text), re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
            return None
            
        print("Extracting pure code blocks...")
        df["pure_code"] = df["answer"].apply(extract_pure_code)
        
        # Drop rows where regex failed to find a standard code block
        df = df.dropna(subset=["pure_code"]).copy()
        
        # 3. Convert back to HuggingFace Dataset for fast tokenizer mapping
        hf_dataset = Dataset.from_pandas(df)
        
        # 4. Initialize tokenizer to filter out massive sequences (>1024 tokens)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        
        def is_within_length(example):
            messages = self.format_prompt(example)["messages"]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            return len(tokenizer.encode(full_text)) <= 1024
            
        print("Filtering for max_seq_length <= 1024...")
        filtered_dataset = hf_dataset.filter(is_within_length, num_proc=4)
        
        # 5. Shuffle and subset to 7,500 to maintain balance with other domains
        shuffled_dataset = filtered_dataset.shuffle(seed=42)
        subset_size = min(7500, len(shuffled_dataset))
        subset_dataset = shuffled_dataset.select(range(subset_size))
        
        # 6. Split 5% for evaluation
        self.dataset = subset_dataset.train_test_split(test_size=0.05, seed=42)
        return self.dataset

    def format_prompt(self, example: dict) -> dict:
        """
        Maps the pure algorithmic task to Llama 3 format.
        Forces the model to act as a strict code engine without conversational filler.
        """
        messages = [
            {
                "role": "system", 
                "content": "You are a strict algorithmic coding engine. Write perfectly self-contained Python code. Output nothing but the code."
            },
            {
                "role": "user", 
                "content": str(example.get("query", "")).strip()
            },
            {
                "role": "assistant", 
                "content": str(example.get("pure_code", "")).strip() # Using the extracted code!
            }
        ]
        return {"messages": messages}