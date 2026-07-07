import os
import sys
import torch
from transformers import AutoTokenizer
from datasets import Dataset
# Fix for Windows DLL loading
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_ops.gsm8k_preprocessor import GSM8KPreprocessor
from src.data_ops.metamath_preprocessor import MetaMathPreprocessor
from src.data_ops.codefeedback_preprocessor import CodeFeedbackPreprocessor
from src.model_ops.lora_manager import LoraManager
from src.training.trainer import AdapterTrainer
from scripts.evaluate_gsm8k import main as gsm8k_main
from scripts.evaluate_mbpp import main as mbpp_main
from dotenv import load_dotenv
from huggingface_hub import login
from src.model_ops.inference_manager import InferenceManager


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
def run_overfit_test():
    print("=== STARTING OVERFIT VERIFICATION ===")
    model_id = "meta-llama/Llama-3.2-3B"
    max_length = 256
    
    print("1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("2. Extracting exactly 2 rows of test data...")
    preprocessor = GSM8KPreprocessor(tokenizer, max_length=max_length, data_size=2)
    raw_dataset = preprocessor.process()
    
    train_data = format_dataset_for_base_model(raw_dataset["train"], tokenizer, max_length)
    
    # Save the raw text of the first question to test generation
    test_question = raw_dataset["train"][0]["prompt"]
    target_answer = raw_dataset["train"][0]["output"]

    print("3. Initializing base model in 4-bit...")
    lora_manager = LoraManager(model_id=model_id)
    base_model = lora_manager.load_base_model()
    peft_model = lora_manager.get_peft_setup(
        base_model, r=32, alpha=64, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # --- BEFORE TRAINING TEST ---
    print(f"\n{'='*50}")
    print(" BEFORE TRAINING INFERENCE")
    print(f"{'='*50}")
    inference_prompt = f"Question: {test_question}\nAnswer:\n"
    inputs = tokenizer(inference_prompt, return_tensors="pt").to(peft_model.device)
    
    peft_model.eval() # Set to eval mode for generation
    with torch.no_grad():
        outputs = peft_model.generate(
            **inputs, 
            max_new_tokens=100, 
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    print(tokenizer.decode(outputs[0], skip_special_tokens=False))
    print(f"{'='*50}\n")

    # --- AGGRESSIVE OVERFIT TRAINING ---
    print("4. FIRING OVERFIT TRAINING LOOP (20 Epochs)...")
    peft_model.train() # Set back to train mode
    trainer = AdapterTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=train_data, # Using train data for eval to bypass crash on 0 size
        output_dir="./models/debug_overfit"
    )

    # We train on 2 rows with batch_size=2 for 20 epochs. 
    # This forces exactly 20 optimization steps. Loss should approach 0.
    trainer.train(
        adapter_name="debug_math_lora",
        batch_size=2,
        grad_accum=1,  
        epochs=20,
        logging_steps=1,
        learning_rate=5e-4, # Higher learning rate to force rapid memorization
        save_steps=50 # Avoid saving checkpoints during this fast run
    )

    # --- AFTER TRAINING TEST ---
    print(f"\n{'='*50}")
    print(" AFTER TRAINING INFERENCE")
    print(f"{'='*50}")
    print(f"TARGET GOAL WAS:\n{target_answer}\n")
    print("MODEL OUTPUT IS:")
    
    peft_model.eval()
    with torch.no_grad():
        outputs_after = peft_model.generate(
            **inputs, 
            max_new_tokens=100, 
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    print(tokenizer.decode(outputs_after[0], skip_special_tokens=False))
    print(f"{'='*50}")
    print("=== OVERFIT DEBUG COMPLETE ===")

def debug_gsm8k_evaluation():
    print("=== STARTING LOCAL EVALUATION DEBUG ===")
    
    # Store the original sys.argv to prevent side effects
    original_argv = sys.argv
    
    # Inject the exact CLI parameters needed for the micro-test
    sys.argv = [
        "evaluate_gsm8k.py", 
        "--base_model_id", "meta-llama/Llama-3.2-3B",
        # "--adapter_path", "./models/adapters/adapter_math/math_gsm8k_main_3b_base_07.06.2026",
        "--domain", "math",
        "--num_samples", "2",  # Restrict to 2 rows to prevent local OOM and save time
        "--max_length", "512",
        "--output_file", "artifacts/generated_results/debug_gsm8k_eval_base_results.json"
    ]
    
    try:
        # Execute the production pipeline
        gsm8k_main()
        print("\n=== EVALUATION DEBUG PASSED ===")
        print("Check 'artifacts/generated_results/debug_gsm8k_eval_results.json' to verify extraction accuracy.")
    except Exception as e:
        print("\n=== EVALUATION DEBUG CRASHED ===")
        raise e
    finally:
        # Safely restore the original system arguments
        sys.argv = original_argv
def debug_mbpp_evaluation():
    print("=== STARTING LOCAL MBPP EVALUATION DEBUG ===")
    
    # Store the original sys.argv to prevent side effects
    original_argv = sys.argv
    
    # Inject the exact CLI parameters needed for the micro-test
    sys.argv = [
        "evaluate_mbpp.py", 
        "--base_model_id", "meta-llama/Llama-3.2-3B",
        # Uncomment and update the line below to test with your CodeFeedback adapter
        # "--adapter_path", "./models/adapters/adapter_code/code_codefeedback_3b_base_07.06.2026",
        "--domain", "code",
        "--num_samples", "2",  # Restrict to 2 rows to prevent local OOM
        "--max_length", "512",
        "--output_file", "artifacts/generated_results/debug_mbpp_eval_base_results.json"
    ]
    
    try:
        # Execute the production pipeline
        mbpp_main()
        print("\n=== MBPP EVALUATION DEBUG PASSED ===")
        print("Check 'artifacts/generated_results/debug_mbpp_eval_base_results.json' to verify generation.")
    except Exception as e:
        print("\n=== MBPP EVALUATION DEBUG CRASHED ===")
        raise e
    finally:
        # Safely restore the original system arguments
        sys.argv = original_argv

def debug_inference_manager():
    print("\n=== STARTING INFERENCE MANAGER VERIFICATION ===")
    model_id = "meta-llama/Llama-3.2-3B"
    raw_adapter_paths = {
        "code": "./models/adapters/adapter_code/code_codefeedback_3b_base_07.06.2026",
        "math": "./models/adapters/adapter_math/math_gsm8k_main_3b_base_07.06.2026",
        "hybrid":""
    }

    # Check which adapters actually exist to prevent PEFT crashes
    valid_adapters = {}
    for domain, path in raw_adapter_paths.items():
        if os.path.exists(path):
            valid_adapters[domain] = path
        else:
            print(f"  [WARNING] Missing adapter for '{domain}' at: {path}")

    if not valid_adapters:
        print("  [WARNING] No valid adapters found. Running in BASE MODEL ONLY mode.")

    print("1. Initializing InferenceManager for base model execution...")
    manager = InferenceManager(base_model_id=model_id, adapter_paths=valid_adapters)
    
    # 2. Test standard base model query routing
    test_query = "What is 5+5 ?"
    print(f"2. Executing text generation under 'general' domain rule...")
    print(f"Input query: {test_query!r}\n")
    
    try:
        # Utilizing max_new_tokens to prevent prolonged generation during local diagnostic loop
        response = manager.generate(
            user_prompt=test_query,
            domain="math",
            stream=True,
            # max_new_tokens=200
        )
        print(f"\n\nResponse completely decoded: {response!r}")
        print("\n=== INFERENCE MANAGER DEBUG PASSED ===")
    except Exception as e:
        print("\n=== INFERENCE MANAGER DEBUG CRASHED ===")
        raise e
def main():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("Error: HF_TOKEN not found in .env file")
    debug_inference_manager()
if __name__ == "__main__":
    main()