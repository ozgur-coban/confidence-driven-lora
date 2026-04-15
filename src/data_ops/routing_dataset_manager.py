import json
import os
from datasets import load_dataset

class DatasetManager:
    def __init__(self, local_paths=dict,
                   domain_configs=dict):
        self.clean_squad_path =local_paths.get("squad")
        self.clean_ultra_path =local_paths.get("ultrachat")
        self.domain_configs = domain_configs


    def _load_json(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cleaned dataset not found at {filepath}. Run the filtering notebook first.")
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_training_data(self, specialist_samples: int = 6000, squad_samples: int = 25000, ultra_samples: int = 25000,general_samples:int=6000) -> dict:
        """Dynamically loads training data based on injected configuration."""
        training_dict = {}

        print(f"Loading Hugging Face Specialist Datasets ({specialist_samples} each)...")
        for domain, config in self.domain_configs.items():
            ds = load_dataset(config["path"], config.get("name"), split=config.get("train_split", "train"))
            # Dynamically grab the correct column (e.g., 'question' or 'query')
            col = config["column"]
            if domain == "general":
                training_dict[domain] = [item[col] for item in ds.select(range(general_samples))]
            else:
                training_dict[domain] = [item[col] for item in ds.select(range(specialist_samples))]

        print(f"Loading Local General Datasets ({squad_samples} SQuAD, {ultra_samples} UltraChat)...")
        squad_data = self._load_json(self.clean_squad_path)
        ultra_data = self._load_json(self.clean_ultra_path)
        if "general" not in training_dict:
            training_dict["general"] = []
        training_dict["general"] += squad_data[:squad_samples] + ultra_data[:ultra_samples]

        return training_dict
    def get_testing_data(self, test_samples:int=500, specialist_offset:int=6000):
        testing_dict={}
        for domain, config in self.domain_configs.items():
            ds = load_dataset(config["path"], config.get("name"), split=config.get("test_split", "test"))
            col = config["column"]
            testing_dict[domain] = [item[col] for item in ds.select(range(specialist_offset, specialist_offset + test_samples))]
        testing_dict["general"] = self._load_json(self.clean_squad_path)[-test_samples:]
        return testing_dict
    
        
