from datasets import load_dataset, DatasetDict
from .base_preprocessor import BasePreprocessor

class MetaMathPreprocessor(BasePreprocessor):
    def load_data(self) -> DatasetDict:
        """Loads, filters by length, samples, and splits the MetaMathQA dataset."""
        print("Downloading MetaMathQA dataset...")
        dataset = load_dataset("meta-math/MetaMathQA", split="train")
        
        # Fast heuristic filter: Keep rows where combined query/response length is under ~1800 characters
        # 1 token is roughly 4 characters. 1800 chars safely approximates < 512 tokens.
        print("Filtering for sequences safely under 512 tokens...")
        filtered_dataset = dataset.filter(
            lambda x: (len(str(x["query"])) + len(str(x["response"]))) < 1800
        )
        
        # Shuffle with a fixed seed and select exactly 15,000 samples from the safe pool
        print("Sampling 15,000 short sequences...")
        dataset = filtered_dataset.shuffle(seed=42).select(range(15000))
        
        # Split into 10k train and 5k remaining
        train_test_split = dataset.train_test_split(test_size=5000, seed=42)
        
        # Split the remaining 5k into 2k validation and 3k test
        val_test_split = train_test_split['test'].train_test_split(test_size=3000, seed=42)
        
        self.dataset = DatasetDict({
            'train': train_test_split['train'],          # 10,000 samples
            'validation': val_test_split['train'],       # 2,000 samples
            'test': val_test_split['test']               # 3,000 samples
        })
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