DOMAIN_REGISTRY = {
    "math": {
        "system": "You are a mathematical reasoning assistant. Solve the problem step-by-step and provide the final answer after ####.",
        "kwargs": {
            "max_new_tokens": 512, 
            "do_sample": False
        }
    },
    "code": {
        "system": "You are an expert software engineer. Write clean, efficient, and well-commented code.",
        "kwargs": {
            "max_new_tokens": 1024, 
            "do_sample": False
        }
    },
    "medical": {
        "system": "You are a highly knowledgeable medical professional. Provide accurate and evidence-based medical information.",
        "kwargs": {
            "max_new_tokens": 512, 
            "do_sample": True, 
            "temperature": 0.3, 
            "top_p": 0.9
        }
    },
    "general": {
        "system": "You are a helpful, conversational AI assistant.",
        "kwargs": {
            "max_new_tokens": 512, 
            "do_sample": True, 
            "temperature": 0.6, 
            "top_p": 0.9
        }
    }
}