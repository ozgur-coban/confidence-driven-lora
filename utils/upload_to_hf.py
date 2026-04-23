import os
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    print("  [AUTH] Hugging Face token found. Logging in...")
    login(token=hf_token)
else:
    print(
        "  [WARNING] HF_TOKEN not found in .env file. Gated models will fail to load."
    )

# 2. Load your local JSON into a Hugging Face Dataset object
print("Loading local JSON...")
dataset = load_dataset(
    "json",
    data_files="./data/train/medical/preprocessed/icliniq_genmed_combined.json",
    split="train",
)

repo_id = "ozgur-coban/medical-lora-10k"

print(f"Pushing to {repo_id}...")
# Setting private=True
dataset.push_to_hub(repo_id, private=True)

print("Upload complete!")
