import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login


def main():
    # 1. Authenticate
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN not found in environment. Please set it before running."
        )

    print("Authenticating with Hugging Face...")
    login(token=hf_token)

    api = HfApi()
    repo_id = "ozgur-coban/confidence-driven-lora"

    # Ensure the repo exists (creates it if you haven't already on the website)
    api.create_repo(repo_id=repo_id, exist_ok=True)
    print(f"\nTarget Repository: https://huggingface.co/{repo_id}\n")

    # ---------------------------------------------------------
    # 2. Upload Router Artifacts (Individual Files)
    # ---------------------------------------------------------
    print("Uploading Router Artifacts...")
    router_files = {
        "./models/semantic_centroids.npy": "router/semantic_centroids.npy",
        "./models/syntax_router.joblib": "router/syntax_router.joblib",
    }

    for local_path, hub_path in router_files.items():
        if os.path.exists(local_path):
            print(f"  -> Uploading {local_path} to {hub_path}...")
            api.upload_file(
                path_or_fileobj=local_path, path_in_repo=hub_path, repo_id=repo_id
            )
        else:
            print(f"  [WARNING] Could not find local router file: {local_path}")

    # ---------------------------------------------------------
    # 3. Upload LoRA Adapters (Entire Folders)
    # ---------------------------------------------------------
    print("\nUploading LoRA Adapters...")

    # Update these local paths to match the exact names of your final training output folders
    adapters = {
        "adapters/code": "./models/adapters/adapter_code/code_codefeedback_8b_instruct_final_22_04_2026",
        "adapters/math": "./models/adapters/adapter_math/math_gsm8k_socratic_8b_instruct_final_22_04_2026",
        "adapters/medical": "./models/adapters/medical_medalpaca_8b_instruct_final_22_04_2026",
    }

    for hub_folder, local_folder in adapters.items():
        if os.path.exists(local_folder):
            print(f"  -> Uploading directory {local_folder} to {hub_folder}...")
            api.upload_folder(
                folder_path=local_folder, path_in_repo=hub_folder, repo_id=repo_id
            )
        else:
            print(f"  [WARNING] Could not find local adapter folder: {local_folder}")

    print("\n✅ Upload Complete! All files are now live on Hugging Face.")


if __name__ == "__main__":
    main()
