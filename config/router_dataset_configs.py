ROUTER_DATASET_CONFIGS = {
    "local_paths": {
        "squad": "data/filtered/clean_squad.json",
        "ultra": "data/filtered/clean_ultrachat.json",
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
            "test_split": "test",  # Or 'validation' depending on the repo
            "column": "query",
        },
        "medical": {
            "path": "GBaker/MedQA-USMLE-4-options",
            "name": None,
            "train_split": "train",
            "test_split": "test",
            "column": "question",
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
