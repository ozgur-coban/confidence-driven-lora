import json
import os
from datasets import load_dataset

class DatasetManager:
    def __init__(self, 
                 clean_squad_path="data/filtered/clean_squad.json", 
                 clean_ultra_path="data/filtered/clean_ultrachat.json"):
        self.clean_squad_path = clean_squad_path
        self.clean_ultra_path = clean_ultra_path

    def _load_json(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cleaned dataset not found at {filepath}. Run the filtering notebook first.")
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_training_data(self, specialist_samples=6000, squad_samples=25000, ultra_samples=25000):
        """
        Builds the training dictionary. 
        Mixes SQuAD (Trivia/Facts) and UltraChat (AI Instructions) for a massive, robust General domain.
        """
        print(f"Loading Hugging Face Specialist Datasets ({specialist_samples} each)...")
        math_ds = load_dataset("openai/gsm8k", "main", split="train")
        code_ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
        med_ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")

        print(f"Loading Local Cleaned General Datasets ({squad_samples} SQuAD, {ultra_samples} UltraChat)...")
        squad_data = self._load_json(self.clean_squad_path)
        ultra_data = self._load_json(self.clean_ultra_path)

        # Safely slice the data
        general_train = squad_data[:squad_samples] + ultra_data[:ultra_samples]

        return {
            "math": [item['question'] for item in math_ds.select(range(specialist_samples))],
            "code": [item['query'] for item in code_ds.select(range(specialist_samples))],
            "medical": [item['question'] for item in med_ds.select(range(specialist_samples))],
            "general": general_train
        }

    def get_testing_data(self, test_samples=500, specialist_offset=6000):
        """
        Builds the testing dictionary.
        Uses pure UltraChat for the General test to simulate realistic user queries.
        Grabs from the END of the arrays to ensure zero data leakage from training.
        """
        math_ds = load_dataset("openai/gsm8k", "main", split="train")
        code_ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
        med_ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")

        ultra_data = self._load_json(self.clean_ultra_path)
        
        # Grab from the very back of the array
        general_test = ultra_data[-test_samples:]

        return {
            "math": [item['question'] for item in math_ds.select(range(specialist_offset, specialist_offset + test_samples))],
            "code": [item['query'] for item in code_ds.select(range(specialist_offset, specialist_offset + test_samples))],
            "medical": [item['question'] for item in med_ds.select(range(specialist_offset, specialist_offset + test_samples))],
            "general": general_test
        }