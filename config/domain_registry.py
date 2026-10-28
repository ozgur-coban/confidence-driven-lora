DOMAIN_REGISTRY = {
    "math": {
        "kwargs": {
            "max_new_tokens": 1024,
            "do_sample": False,
            "repetition_penalty": 1.05,
        },
    },
    "code": {
        "kwargs": {
            "max_new_tokens": 2048,
            "do_sample": True,
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "repetition_penalty": 1.1,
        },
    },
    "hybrid": {
        "kwargs": {
            "max_new_tokens": 2048,
            "do_sample": True,
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "repetition_penalty": 1.1,
        },
    },
    "medical": {
        "kwargs": {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.1,
            "top_p": 0.90,
            "repetition_penalty": 1.1,
        },
    },
    "general": {
        "kwargs": {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.5,
            "top_p": 0.90,
            "repetition_penalty": 1.05,
        },
    },
}