import torch

# 1. Target the specific bitsandbytes class before importing transformers
try:
    import bitsandbytes as bnb
    # This reaches into the 4-bit parameter class and fixes its constructor
    original_params4bit_new = bnb.nn.Params4bit.__new__
    
    def patched_params4bit_new(cls, *args, **kwargs):
        # Remove the 'annoying' keyword that Transformers 4.48+ sends
        kwargs.pop('_is_hf_initialized', None)
        return original_params4bit_new(cls, *args, **kwargs)
    
    bnb.nn.Params4bit.__new__ = patched_params4bit_new
    print("SUCCESS: BitsAndBytes 4-bit patch applied.")
except Exception as e:
    print(f"Patch failed or not needed: {e}")

# 2. Now import everything else
import time
import statistics
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def test_inference():
    model_id = "NousResearch/Meta-Llama-3-8B-Instruct"
    
    print("1. Initializing 4-bit configuration...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True
    )
    device_map = {
        "model.embed_tokens": 0,
        "model.norm": 0,
        "lm_head": 0
    }
    print("2. Loading Base Model to GPU...")
    
    # Map layers 0-15 to GPU, 16-31 to CPU
    for i in range(32):
        if i < 14:
            device_map[f"model.layers.{i}"] = 0
        else:
            device_map[f"model.layers.{i}"] = "cpu"

    print("2. Loading Model with Manual Device Map...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map=device_map,        # Use our manual map
        offload_folder="offload_temp",
        torch_dtype=torch.float16,    # float16 is safer for CPU offloading
        low_cpu_mem_usage=True
    )

    print("3. Model Loaded! Testing Generation...")
    test_questions = [
        "What is 15 percent of 80?",
        "If a train travels at 60mph, how far does it go in 2.5 hours?",
        "Solve for x: 2x + 10 = 26",
        "What is the square root of 144?",
        "A shirt costs $20 and is on 25% sale. What is the new price?",
        "What is 7 multiplied by 8, then plus 12?",
        "If I have 3 apples and buy 5 more, then give 2 away, how many do I have?",
        "What is the area of a rectangle with length 5 and width 3?",
        "How many minutes are in 4.5 hours?",
        "What is 1000 divided by 8?"
    ]

    durations = []
    print(f"\n--- Starting Benchmark (10 Examples) ---\n")

    for i, question in enumerate(test_questions):
        # Format with the specific system prompt from your training data
        messages = [
            {"role": "system", "content": "You are a mathematical reasoning assistant. Solve the problem step-by-step and provide the final answer after ####."},
            {"role": "user", "content": question}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100, # Increased slightly to allow for step-by-step reasoning
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
        end_time = time.time()
        
        elapsed = end_time - start_time
        durations.append(elapsed)

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        print(f"Example {i+1}:")
        print(f"Q: {question}")
        print(f"A: {response.strip()}")
        print(f"Time: {elapsed:.2f}s")
        print("-" * 30)

        # Clear VRAM cache to prevent fragmentation
        torch.cuda.empty_cache()

    # 5. Summary Statistics
    print("\n--- Benchmark Summary ---")
    print(f"Total Time: {sum(durations):.2f}s")
    print(f"Average Time per Example: {statistics.mean(durations):.2f}s")
    print(f"Fastest: {min(durations):.2f}s")
    print(f"Slowest: {max(durations):.2f}s")

if __name__ == "__main__":
    test_inference()

