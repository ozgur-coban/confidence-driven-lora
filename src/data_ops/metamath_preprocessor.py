from datasets import load_dataset, DatasetDict
from .base_preprocessor import BasePreprocessor

class MetaMathPreprocessor(BasePreprocessor):
    def __init__(self, tokenizer, max_length=512, data_size=15000):
        super().__init__(tokenizer, max_length)
        self.data_size=data_size

    def load_data(self) -> DatasetDict:
        """Loads, filters by length, samples, and splits the MetaMathQA dataset."""
        print("Downloading MetaMathQA dataset...")
        dataset = load_dataset("meta-math/MetaMathQA", split="train")
        def is_within_length(example):
            messages=self.format_prompt(example)
            full_text=self.tokenizer.apply_chat_template(messages,tokenize=False)
            return len(self.tokenizer.encode(full_text)) <= self.max_length
        
        filtered_dataset=dataset.filter(is_within_length,num_proc=4)
        print(f"Dropped {len(dataset) - len(filtered_dataset)} oversized rows.")
        
        # Shuffle with a fixed seed and select exactly 15,000 samples from the safe pool
        subset_size=min(self.data_size,len(filtered_dataset))
        print(f"Sampling {subset_size} short sequences...")
        shuffled_dataset = filtered_dataset.shuffle(seed=42)
        subset_dataset = shuffled_dataset.select(range(subset_size))
        
        # Split into 10k train and 5k remaining
        train_test_split = subset_dataset.train_test_split(test_size=5000, seed=42)
        
        # Split the remaining 5k into 2k validation and 3k test
        val_test_split = train_test_split['test'].train_test_split(test_size=3000, seed=42)
        
        self.shuffled_dataset = DatasetDict({
            'train': train_test_split['train'],          # 10,000 samples
            'validation': val_test_split['train'],       # 2,000 samples
            'test': val_test_split['test']               # 3,000 samples
        })
        return self.shuffled_dataset

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
                "content": "You are a mathematical reasoning assistant. Solve the problem step-by-step and provide the final answer after ####."
            },
            {
                "role": "user",
                "content": example["query"]
            },
            {
                "role": "assistant",
                "content": standardized_ans
            }
        ]
        return {"messages": messages}