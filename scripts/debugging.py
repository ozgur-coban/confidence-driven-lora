import os
import sys
from transformers import AutoTokenizer
from datasets import Dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_ops.gsm8k_preprocessor import GSM8KPreprocessor
from src.data_ops.metamath_preprocessor import MetaMathPreprocessor
from src.data_ops.codefeedback_preprocessor import CodeFeedbackPreprocessor

def format_dataset_for_base_model(dataset, tokenizer, max_length=512,is_verbose=False):
    """The exact function from your train_adapter.py script."""
    def map_format(example):
        q = example.get("prompt") or ""
        a = example.get("output") or ""
        
        prompt_text = f"Question: {q}\nAnswer:\n"
        answer_text = f"{a}{tokenizer.eos_token}"
        if(is_verbose):
            print(f"prompt_text: {prompt_text}")
            print(f"answer_text: {answer_text}")
        
        prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids) 
        }

    # num_proc=1 used here just for local debugging
    return dataset.map(map_format, remove_columns=dataset.column_names, num_proc=1)

def test_tokenizer():
    model_id = "meta-llama/Llama-3.2-3B" 
    tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False)

    q = "Write a python script to add two numbers."
    a = "def add(a, b):\n    return a + b"

    # 1. Create a dummy Hugging Face Dataset (mimicking your preprocessors)
    dummy_data = {"prompt": [q], "output": [a]}
    raw_dataset = Dataset.from_dict(dummy_data)
    
    print(f"Original Dataset Columns: {raw_dataset.column_names}")

    # 2. Run the mapping function
    processed_dataset = format_dataset_for_base_model(raw_dataset, tokenizer)
    
    print(f"Processed Dataset Columns: {processed_dataset.column_names}\n")

    # 3. Extract the arrays for verification
    row = processed_dataset[0]
    input_ids = row["input_ids"]
    labels = row["labels"]
    attention_mask = row["attention_mask"]

    # 4. Print the alignment visualization
    print(f"{'Text Token':<25} | {'Input ID':<10} | {'Label ID':<10} | {'Attn Mask':<10}")
    print("-" * 65)
    for i in range(len(input_ids)):
        token_str = repr(tokenizer.decode([input_ids[i]]))
        print(f"{token_str:<25} | {input_ids[i]:<10} | {labels[i]:<10} | {attention_mask[i]:<10}")
def test_real_dataset(dataset_preprocessor,is_verbose=False):
    model_id = "meta-llama/Llama-3.2-3B" 
    tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Initialize a real preprocessor with a tiny dataset size for fast testing
    print("Initializing Preprocessor...")
    
    # You can swap this out for MetaMathPreprocessor or CodeFeedbackPreprocessor
    preprocessor = dataset_preprocessor(tokenizer, max_length=512, data_size=10)
    
    # 2. Run the extraction pipeline
    raw_dataset_dict = preprocessor.process()
    train_data = raw_dataset_dict["train"]
    
    print(f"\nExtracted Dataset Columns: {train_data.column_names}")
    print(f"Sample Prompt Preview: {train_data[0]['prompt'][:50]}...\n")

    # 3. Run the mapping function
    processed_dataset = format_dataset_for_base_model(train_data, tokenizer,is_verbose=is_verbose)
    
    print(f"Processed Dataset Columns: {processed_dataset.column_names}\n")

    # 4. Extract the arrays for verification
    row = processed_dataset[0]
    input_ids = row["input_ids"]
    labels = row["labels"]
    attention_mask = row["attention_mask"]
    # 5. Print the alignment visualization
    print(f"{'Text Token':<25} | {'Input ID':<10} | {'Label ID':<10} | {'Attn Mask':<10}")
    print("-" * 65)
    
    # Print the first 80 tokens to catch the prompt -> answer transition
    for i in range(len(input_ids)):
        token_str = repr(tokenizer.decode([input_ids[i]]))
        print(f"{token_str:<25} | {input_ids[i]:<10} | {labels[i]:<10} | {attention_mask[i]:<10}")
    
def main():
    test_real_dataset(GSM8KPreprocessor,is_verbose=True)
    test_real_dataset(MetaMathPreprocessor,is_verbose=True)
    test_real_dataset(CodeFeedbackPreprocessor,is_verbose=True)


if __name__ == "__main__":
    main()