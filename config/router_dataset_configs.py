ROUTER_DATASET_CONFIGS = {
    "local_paths": {
        "squad": "data/filtered/clean_squad.json",
        "ultrachat": "data/filtered/clean_ultrachat.json",
    },
    "domain_rules": {
        "math": {
            "path": "openai/gsm8k",
            "name": "main",
            "train_split": "train",
            "test_split": "test",
            "column": "question",
        },
        "code": {
            "path": "m-a-p/CodeFeedback-Filtered-Instruction",
            "name": None,
            "train_split": "train",
            "test_split": "test",
            "column": "query",
        },
        "medical": {
            "path": "GBaker/MedQA-USMLE-4-options",
            "name": None,
            "train_split": "train",
            "test_split": "test",
            "column": "question",
        },
        "hybrid": {
            "path": "ozgur-coban/code_math_hybrid_300",  # Works for 100 or 500 once updated
            "name": None,
            "train_split": "train",
            "test_split": "train",  # Fallback to train since you won't have a specific test split uploaded
            "column": "query",
        },
        "general": {
            "path": "rajpurkar/squad_v2",
            "name": None,
            "train_split": "train",
            "test_split": "validation",
            "column": "question",
        },
    },
}