import re

# --- MATH EVALUATION (From your provided script) ---
def clean_math_string(s: str) -> str:
    """Normalizes math outputs: safely unwraps \\boxed{}, strips $, cleans commas."""
    if not isinstance(s, str):
        return ""
        
    s = s.strip()
    s = re.sub(r"^#+", "", s)
    s = re.sub(r"#+$", "", s)
    s = s.strip()

    if "\\boxed{" in s:
        start = s.find("\\boxed{") + 7
        depth = 1
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
            if depth == 0:
                s = s[start:i]
                break

    s = s.strip("$ \n\t.")
    s = re.sub(r"^(The )?(final )?answer is\s*:?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^Therefore,?\s*(the )?.*?\s+is\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^So,?\s*(the )?.*?\s+is\s*", "", s, flags=re.IGNORECASE)
    s = s.strip("$ \n\t.")

    if len(s) > 10 and not any(op in s for op in ["+", "-", "*", "/", "\\", "^", "=", "x", "y", "z"]):
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", s)
        if nums:
            s = nums[-1]

    if re.match(r"^-?[\d,]+(?:\.\d+)?$", s):
        s = s.replace(",", "")

    return s.strip()


def extract_model_answer(text: str) -> str:
    """Robust extraction handling ####, common phrases, and raw number fallbacks."""
    if not isinstance(text, str):
        return "Extraction Failed"
        
    ans = "Extraction Failed"
    if "####" in text:
        parts = text.split("####")
        ans = parts[1] if len(parts) > 1 and parts[1].strip() else parts[-1]
    elif "The final answer is:" in text:
        ans = text.split("The final answer is:")[-1]
    elif "The answer is:" in text:
        ans = text.split("The answer is:")[-1]
    else:
        numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
        if numbers:
            ans = numbers[-1]

    if ans == "Extraction Failed":
        return ans
    return clean_math_string(ans)


def evaluate_math(response: str, ground_truth: str) -> bool:
    """Extracts answers and compares them strictly."""
    extracted_ans = extract_model_answer(response)
    truth_ans = extract_model_answer(ground_truth)
    
    if extracted_ans == "Extraction Failed" or truth_ans == "Extraction Failed":
        return False
        
    try:
        return float(extracted_ans) == float(truth_ans)
    except ValueError:
        return extracted_ans == truth_ans
