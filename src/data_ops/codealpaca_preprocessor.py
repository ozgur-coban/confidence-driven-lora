from datasets import load_dataset, DatasetDict
from .base_preprocessor import BasePreprocessor

class CodeAlpacaPreprocessor(BasePreprocessor):
    def load_data(self) -> DatasetDict:
        raw_dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        
        # 1. High-Quality Filter: Only keep rows with clear code markers
        code_signals = [
            "def ", "import ", "class ", "return ", "print(", "{", "const ", "var ",
            "<html>", "SELECT ", "FROM ", "#include", "public ", "static ", "void ",
            "if ", "for ", "while ", "let ", "arr =", "lst =", " = "
        ]
        
        def has_code(example):
            output = str(example["output"])
            return any(kw in output for kw in code_signals)
            
        filtered_dataset = raw_dataset.filter(has_code)
        
        # 2. Shuffle and subset to match GSM8K volume (7,500 rows)
        subset = filtered_dataset.shuffle(seed=42).select(range(7500))
        
        # 3. Final Split
        self.dataset = subset.train_test_split(test_size=0.05, seed=42)
        return self.dataset

    def format_prompt(self, example: dict) -> dict:
        user_content = str(example.get("instruction", "")).strip()
        raw_input = str(example.get("input", "")).strip()
        
        if raw_input and raw_input.lower() != "< noinput >":
            user_content += f"\n\nInput:\n{raw_input}"

        messages = [
            {
                "role": "system", 
                "content": "You are an expert programming assistant. Write clean, efficient, and well-documented code."
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": str(example.get("output", "")).strip()}
        ]
        return {"messages": messages}