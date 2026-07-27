ROUTER_DATASET_CONFIGS = {
    "local_paths": {
        "squad": "data/filtered/clean_squad.json",
        "ultrachat": "data/filtered/clean_ultrachat.json",
        "hybrid_train_local": "./data/gsm8k_hybrid_train.json",
        "hybrid_test_local": "./data/gsm8k_hybrid_test.json",
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
        "hybrid": {
            "path": "ozgur-coban/code_math_hybrid_300",
            "name": None,
            "train_split": "train",
            "test_split": "train",
            "column": "query",
            "local_train": "./data/gsm8k_hybrid_train.json",
            "local_test": "./data/gsm8k_hybrid_test.json",
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