DOMAIN_REGISTRY = {
    "math": {
        "training_prompt": "The following is a step-by-step mathematical solution. All logic is explained clearly, and the final answer is provided at the very end after ####.\n\nQuestion: {query}\n\nAnswer: {response}{eos}",
        "template": "The following is a step-by-step mathematical solution. All logic is explained clearly, and the final answer is provided at the very end after ####.\n\nQuestion: {query}\n\nAnswer: ",
        "kwargs": {
            "max_new_tokens": 1024, 
            "do_sample": False,
            "repetition_penalty": 1.05
        }
    },
    "code": {
        "training_prompt": "The following is clean, efficient, and well-documented programming logic written by a senior software engineer.\n\nRequirement: {query}\n\nSolution: {response}{eos}",
        "template": "The following is clean, efficient, and well-documented programming logic written by a senior software engineer.\n\nRequirement: {query}\n\nSolution: ",
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
        "training_prompt": "The following is a highly accurate, evidence-based clinical analysis and medical assessment.\n\nCase/Query: {query}\n\nClinical Assessment: {response}{eos}",
        "template": "The following is a highly accurate, evidence-based clinical analysis and medical assessment.\n\nCase/Query: {query}\n\nClinical Assessment: ",
        "kwargs": {
            "max_new_tokens": 1024, 
            "do_sample": True, 
            "temperature": 0.1, 
            "top_p": 0.90,
            "repetition_penalty": 1.1
        }
    },
    "general": {
        "training_prompt": "The following is a transcript of an interactive and helpful conversation between a Human and a knowledgeable Assistant.\n\nHuman: {query}\n\nAssistant: {response}{eos}",
        "template": "The following is a transcript of an interactive and helpful conversation between a Human and a knowledgeable Assistant.\n\nHuman: {query}\n\nAssistant: ",
        "kwargs": {
            "max_new_tokens": 1024, 
            "do_sample": True, 
            "temperature": 0.5, 
            "top_p": 0.90,
            "repetition_penalty": 1.05
        }
    }
}