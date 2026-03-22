DOMAIN_REGISTRY = {
    "math": {
        "system": "You are an expert mathematical reasoning assistant. You must solve the problem step-by-step. Show all your work, explain your logic clearly, and provide the final answer at the very end after ####.",
        "kwargs": {
            "max_new_tokens": 1024, 
            "do_sample": False,
            "repetition_penalty": 1.05
        }
    },
    "code": {
        "system": "You are a senior software engineer. Write clean, efficient, and well-documented code. Think through the logic step-by-step before providing the final code block.",
        "kwargs": {
            "max_new_tokens": 2048, 
            "do_sample": True,
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "repetition_penalty": 1.1
        }
    },
    "medical": {
        "system": "You are an expert clinical diagnostician. Provide highly accurate, evidence-based medical information. Analyze the patient's symptoms or medical query systematically before concluding.",
        "kwargs": {
            "max_new_tokens": 1024, 
            "do_sample": True, 
            "temperature": 0.1, 
            "top_p": 0.90,
            "repetition_penalty": 1.1
        }
    },
    "general": {
        "system": "You are a helpful and intelligent AI assistant. Provide clear, concise, and direct answers.",
        "kwargs": {
            "max_new_tokens": 1024, 
            "do_sample": True, 
            "temperature": 0.5, 
            "top_p": 0.90,
            "repetition_penalty": 1.05
        }
    }
}