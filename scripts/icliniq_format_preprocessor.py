import json
import random


def create_medical_dataset(icliniq_path, genmed_path, output_path, sample_size=5000):
    # 1. Load and process iCliniq data
    print(f"Loading iCliniq data from {icliniq_path}...")
    with open(icliniq_path, "r", encoding="utf-8") as f:
        icliniq_raw = json.load(f)

    processed_icliniq = []

    # Using your upgraded, highly-constrained system instruction
    standard_instruction = "You are an expert clinical diagnostician. Provide highly accurate, evidence-based medical information. Analyze the patient's symptoms or medical query systematically before concluding."

    for item in icliniq_raw:
        if "input" in item and "answer_chatgpt" in item:
            processed_icliniq.append(
                {
                    "instruction": standard_instruction,
                    "input": item["input"],
                    "output": item["answer_chatgpt"],
                }
            )

    # 2. Randomly sample 5,000 from the processed iCliniq data
    actual_sample_size = min(sample_size, len(processed_icliniq))
    sampled_icliniq = random.sample(processed_icliniq, actual_sample_size)
    print(f"Sampled {len(sampled_icliniq)} records from iCliniq.")

    # 3. Load and process GenMedGPT data
    print(f"Loading GenMedGPT data from {genmed_path}...")
    with open(genmed_path, "r", encoding="utf-8") as f:
        genmed_data = json.load(f)

    # We must also overwrite the weak instruction in GenMedGPT with your new one
    for item in genmed_data:
        item["instruction"] = standard_instruction

    print(f"Loaded and updated {len(genmed_data)} records from GenMedGPT.")

    # 4. Combine and shuffle the datasets
    combined_dataset = genmed_data + sampled_icliniq
    random.shuffle(combined_dataset)

    # 5. Export the final dataset
    print(
        f"Saving combined dataset ({len(combined_dataset)} records) to {output_path}..."
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_dataset, f, indent=4, ensure_ascii=False)

    print("Dataset generation complete!")


if __name__ == "__main__":
    ICLINIQ_FILE = "./data/train/medical/raw/iCliniq.json"
    GENMED_FILE = "./data/train/medical/raw/GenMedGPT-5k.json"
    OUTPUT_FILE = "./data/train/medical/preprocessed/icliniq_genmed_combined.json"

    create_medical_dataset(ICLINIQ_FILE, GENMED_FILE, OUTPUT_FILE)
