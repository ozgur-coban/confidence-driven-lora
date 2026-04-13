from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from .base_preprocessor import BasePreprocessor

class MedQAPreprocessor(BasePreprocessor):
    def __init__(self, tokenizer: AutoTokenizer, max_length: int=512,data_size:int=7500):
        super().__init__(tokenizer, max_length)
        self.data_size=data_size
        
    def load_data(self) -> DatasetDict:
        """Loads MedQA USMLE, filters by strict token length, shuffles, and subsets."""
        raw_dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
        
        
        # 2. Filter logic: Drop anything that exceeds 512 tokens
        def is_within_length(example):
            messages = self.format_prompt(example)["messages"]
            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            return len(self.tokenizer.encode(full_text)) <= self.max_length
            
        print(f"Filtering MedQA dataset for max_seq_length <={self.max_length}")
        filtered_dataset = raw_dataset.filter(is_within_length, num_proc=4)
        print(f"Dropped {len(raw_dataset) - len(filtered_dataset)} oversized rows.")
        
        # 3. Shuffle and take subset (handling case if filtered data is < 7500)
        shuffled_dataset = filtered_dataset.shuffle(seed=42)
        subset_size = min(self.data_size, len(shuffled_dataset))
        subset_dataset = shuffled_dataset.select(range(subset_size))
        
        # 4. Split 5% for evaluation
        self.dataset = subset_dataset.train_test_split(test_size=0.05, seed=42)
        return self.dataset

    def format_prompt(self, example: dict) -> dict:
        """
        Maps MedQA to Llama 3 format.
        Forces the model to output ONLY the correct letter to allow for exact-match evaluation.
        """
        question = str(example.get("question", "")).strip()
        options = example.get("options", {})
        
        options_list = []
        for letter in ["A", "B", "C", "D"]:
            if letter in options:
                options_list.append(f"{letter}) {options[letter]}")
        options_str = "\n".join(options_list)
        
        input_text = f"{question}\n\nOptions:\n{options_str}"
        
        messages = [
            {
                "role": "system", 
                "content": "You are a medical expert taking the USMLE board exam. Read the clinical vignette and options carefully. Output ONLY the single letter of the correct answer (A, B, C, or D). Do not provide any explanation."
            },
            {
                "role": "user", 
                "content": input_text
            },
            {
                "role": "assistant", 
                "content": str(example.get("answer_idx", "")).strip()
            }
        ]
        return {"messages": messages}