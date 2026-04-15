from datasets import load_dataset, DatasetDict
from .base_preprocessor import BasePreprocessor

class MedicalMeadowFlashcardsPreprocessor(BasePreprocessor):
    def load_data(self) -> DatasetDict:
        """Loads Medical Meadow, shuffles, and subsets to match other adapters."""
        raw_dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
        
        # Shuffle and take 7,500 rows to keep balanced with Math and Code
        shuffled_dataset = raw_dataset.shuffle(seed=42)
        subset_dataset = shuffled_dataset.select(range(7500))
        
        # Split 5% for evaluation
        split_data = subset_dataset.train_test_split(test_size=0.05, seed=42)
        self.dataset = DatasetDict({
            'train': split_data['train'],
            'validation': split_data['test']})
        return self.dataset

    def format_prompt(self, example: dict) -> dict:
        """
        Maps medical flashcards to Llama 3 format.
        CRITICAL: We use 'input' as the user content because 'instruction' is a static placeholder.
        """
        messages = [
            {
                "role": "system", 
                "content": "You are a professional medical assistant. Provide accurate, concise, and helpful medical information."
            },
            {
                "role": "user", 
                "content": str(example.get("input", "")).strip() # The REAL question is here
            },
            {
                "role": "assistant", 
                "content": str(example.get("output", "")).strip()
            }
        ]
        return {"messages": messages}