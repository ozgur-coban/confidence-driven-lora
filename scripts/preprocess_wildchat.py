import json

def preprocess_wildchat_jsonl(input_path, output_path):
    dropped_explicitly_non_english = 0
    kept_conversations = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip():
                continue
                
            chat = json.loads(line)
            conversation = chat.get("conversation", [])
            
            # --- 1. THE FORGIVING LANGUAGE FILTER ---
            is_valid_chat = True
            for msg in conversation:
                lang = msg.get("language")
                
                # If language is explicitly named, and it is NOT English or a null-type string, drop it.
                if lang and isinstance(lang, str):
                    lang_clean = lang.strip().lower()
                    if lang_clean and lang_clean not in ["english", "unknown", "not_detected", "none"]:
                        is_valid_chat = False
                        break  # Stop checking this conversation
                        
            if not is_valid_chat:
                dropped_explicitly_non_english += 1
                continue
                
            # --- 2. FIELD STRIPPING ---
            # Create a lightweight dict containing only our target fields
            cleaned_chat = {
                "conversation_hash": chat.get("conversation_hash"),
                "model": chat.get("model"),
                "hashed_ip": chat.get("hashed_ip"),  # Kept to track unique users
                "conversation": []
            }
            
            for msg in conversation:
                cleaned_chat["conversation"].append({
                    "role": msg.get("role"),
                    "content": msg.get("content")
                })
                
            # Write the compressed JSON object
            outfile.write(json.dumps(cleaned_chat) + "\n")
            kept_conversations += 1

    print("Preprocessing complete!")
    print(f"Dropped (Explicitly Non-English): {dropped_explicitly_non_english}")
    print(f"Kept (English or Unknown): {kept_conversations}")

def is_junk_prompt(text):
    """
    Checks if a user prompt contains unwanted domains or copy-pasted code.
    Returns True if the prompt should be dropped.
    """
    text_lower = text.lower()
    
    # 1. Filter out unwanted topics/domains
    unwanted_keywords = ["image", "blender", "bpy"] # 'bpy' is Blender's python library
    if any(kw in text_lower for kw in unwanted_keywords):
        return True
        
    # 2. Filter out explicit code syntax (copy-pasted scripts)
    code_indicators = [
        "import numpy",     # Raw import statement (e.g., 'import numpy')
        "from numpy",
        "import math",
        "def ",        # Python function definition
        "__main__",     # Standard python execution block
        "#include",
        "<?php",
        "Mozilla/5.0",
        "#!/bin/bash",
    ]
    if any(indicator in text_lower for indicator in code_indicators):
        return True
        
    return False
def extract_prompts_from_cleaned_jsonl(input_path, output_path):
    """
    Extracts raw user prompts from the cleaned JSONL file where the 
    assistant's response contains 'import math', while strictly filtering
    out user prompts that contain code or unwanted topics.
    """
    extracted = []
    dropped_due_to_junk = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            if not line.strip():
                continue
                
            chat = json.loads(line)
            conversation = chat.get("conversation", [])
            hashed_ip = chat.get("hashed_ip")
            conv_hash = chat.get("conversation_hash")
            
            for i, msg in enumerate(conversation):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "").lower()
                    
                    if "import math" in content:
                        if i > 0 and conversation[i-1].get("role") == "user":
                            user_prompt = conversation[i-1].get("content", "").strip()
                            
                            # Apply length filter AND our new heuristic filter
                            if len(user_prompt) > 20:
                                if is_junk_prompt(user_prompt):
                                    dropped_due_to_junk += 1
                                else:
                                    extracted.append({
                                        "prompt": user_prompt,
                                        "hashed_ip": hashed_ip,
                                        "conversation_hash": conv_hash
                                    })
                            break  # Move to the next conversation
                                
    # Save the filtered raw data
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(extracted, outfile, indent=4)
        
    print(f"Extraction complete.")
    print(f"Dropped {dropped_due_to_junk} prompts for containing code/unwanted keywords.")
    print(f"Saved {len(extracted)} clean, conversational prompts to '{output_path}'.")
    
    return extracted

def deduplicate_prompts(prompts_list):
    """
    Deduplicates the list of extracted prompts based on exact prompt text.
    """
    seen_text = set()
    unique_prompts = []
    
    for item in prompts_list:
        text = item["prompt"]
        if text not in seen_text:
            seen_text.add(text)
            unique_prompts.append(item)
            
    print(f"Deduplication complete: {len(unique_prompts)} unique prompts remain.")
    return unique_prompts
def filter_length_outliers(prompts_list, top_percent_to_cut=5):
    """
    Removes the top X% longest prompts (outliers) based on character length.
    Small queries are intentionally preserved.
    """
    if not prompts_list:
        return []

    # Calculate lengths of all prompts and sort them
    lengths = [len(item["prompt"]) for item in prompts_list]
    lengths.sort()
    
    # Determine the index for the cutoff threshold
    # e.g., 95% of 460 items = 437 items to keep
    keep_count = int(len(lengths) * (1 - (top_percent_to_cut / 100.0)))
    
    if keep_count == 0:
        return []
        
    # The cutoff length is the character count of the largest item we want to keep
    cutoff_length = lengths[keep_count - 1]
    
    filtered_prompts = []
    dropped_count = 0
    
    for item in prompts_list:
        # Keep everything up to the cutoff threshold
        if len(item["prompt"]) <= cutoff_length:
            filtered_prompts.append(item)
        else:
            dropped_count += 1
            
    print(f"Length Threshold (Top {top_percent_to_cut}% cut): {cutoff_length} characters")
    print(f"Dropped {dropped_count} outlier prompts that were too long.")
    print(f"Remaining prompts: {len(filtered_prompts)}")
    
    return filtered_prompts

def save_for_gemini(prompts, output_path):
    """
    Formats and saves the final deduplicated prompts to a text file for the Gemini Web UI.
    """
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for idx, item in enumerate(prompts):
            # Replace newlines so each prompt stays on a single numbered line
            clean_prompt = item["prompt"].replace('\n', '  ').replace('\r', '  ')
            outfile.write(f"[{idx}] {clean_prompt}\n")
            
    print(f"Saved {len(prompts)} formatted prompts to '{output_path}'.")


if __name__ == "__main__":
    # 1. Extract and save the raw prompts + metadata to a JSON file
    raw_prompts = extract_prompts_from_cleaned_jsonl(
        input_path="./data/wildchat/wildchat_math_cleaned.jsonl", 
        output_path="./data/wildchat/wildchat_extracted_raw_prompts_cleanjunk.jsonl"
    )
    
    # 2. Deduplicate the raw prompts
    unique_prompts = deduplicate_prompts(raw_prompts)
    sized_prompts = filter_length_outliers(unique_prompts, top_percent_to_cut=5)
    
    # 3. Save the final formatted text file for the Gemini Web UI
    save_for_gemini(
        prompts=sized_prompts, 
        output_path="./data/wildchat/prompts_for_gemini.txt"
    )
    pass