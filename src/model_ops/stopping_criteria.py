import torch
from transformers import StoppingCriteria

class KeywordStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words: list, tokenizer):
        self.stop_words = stop_words
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Decode the last 15 tokens to check for tripwires efficiently.
    
        generated_text = self.tokenizer.decode(input_ids[0][-15:], skip_special_tokens=True)
        
        for word in self.stop_words:
            if word in generated_text:
                return True # Trigger the stop
        return False