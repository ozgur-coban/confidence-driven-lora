import os
import sys
import time
import torch
import warnings

# # Suppress benign framework alerts for clean console execution
# warnings.filterwarnings("ignore")
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# # Ensure the root directory is in the path to resolve your custom src imports
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_ops.inference_manager import InferenceManager


def setup_pipeline_infrastructure(base_model_id, math_path, code_path):
    print(f"\n--- Booting Base Model & Adapters via InferenceManager ---")

    valid_adapters = {"math": math_path, "code": code_path}

    manager = InferenceManager(
        base_model_id=base_model_id,
        adapter_paths=valid_adapters,
        use_few_shot_prompt=False,
    )

    return manager.model, manager.tokenizer


def merge_and_activate(
    model, merge_method, weights, adapter_name="test_merge", density=0.2
):
    print(
        f"\n--- Fusing via {merge_method.upper()} | Weights: math: {weights[0]}, code: {weights[1]} ---"
    )

    if adapter_name in model.peft_config:
        model.delete_adapter(adapter_name)

    t0 = time.time()
    # Uses the native 'math' and 'code' keys instantiated by the InferenceManager
    model.add_weighted_adapter(
        adapters=["math", "code"],
        weights=weights,
        adapter_name=adapter_name,
        combination_type=merge_method,
        density=density,
    )
    merge_time = time.time() - t0
    model.set_adapter(adapter_name)
    print(f"Merge complete and active in {merge_time:.2f} seconds")
    return model


def run_evaluation(model, tokenizer, dataset):
    print("\n--- Running Evaluation ---")
    model.eval()
    for item in dataset:
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        print(f"\nTask Type: {item['type']}")
        print(f"Prompt: {item['prompt']}")
        print(f"Output: {response.strip()}")
        print("-" * 30)


if __name__ == "__main__":
    BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    FIRST_LORA = "./models/adapters/adapter_math/math_gsm8k_socratic_8b_instruct_final_22_04_2026"
    SECOND_LORA = (
        "./models/adapters/adapter_code/code_codefeedback_8b_instruct_final_22_04_2026"
    )

    # 1. Unified Initialization
    model, tokenizer = setup_pipeline_infrastructure(
        BASE_MODEL, FIRST_LORA, SECOND_LORA
    )

    test_dataset = [
        # {
        #     "type": "pure_math",
        #     "prompt": "A bakery sells cupcakes for $3.50 each and brownies for $2.25 each. If a customer buys 4 cupcakes and 6 brownies, and pays with a $50 bill, how much change should they receive? Think step-by-step.",
        # },
        # {
        #     "type": "pure_code",
        #     "prompt": "Write a Python function to find the nth Fibonacci number using dynamic programming (memoization). Return only the function definition.",
        # },
        {
            "type": "hybrid_math_code",
            "prompt": "Write a Python script to calculate the exact probability of rolling a sum greater than 15 using three standard six-sided dice. The script must mathematically evaluate all possible dice combinations, count the successful outcomes, and print the final probability as a reduced fraction.",
        },
        {
            "type": "hybrid_math_code",
            "prompt": "A rubber ball is dropped from a height of 100 meters. Each time it hits the ground, it rebounds to exactly 75% of its previous height. Write a Python function that calculates the total cumulative vertical distance the ball has traveled the exact moment it hits the ground for the 10th time. The function must return the final distance as a float.",
        },
    ]

    # 2. Test Math Adapter Solo
    print("\n>> ACTIVATING: Math Adapter (Solo)")
    model.set_adapter("math")
    run_evaluation(model, tokenizer, test_dataset)

    # 3. Test Code Adapter Solo
    print("\n>> ACTIVATING: Code Adapter (Solo)")
    model.set_adapter("code")
    run_evaluation(model, tokenizer, test_dataset)

    # 4. Test Linear Merge
    model = merge_and_activate(model, merge_method="linear", weights=[0.5, 0.5])
    run_evaluation(model, tokenizer, test_dataset)

    # 5. Test TIES Merge
    model = merge_and_activate(model, merge_method="ties", weights=[0.6, 0.4])
    run_evaluation(model, tokenizer, test_dataset)
