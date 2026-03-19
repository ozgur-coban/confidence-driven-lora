import os
import numpy as np
from src.router.semantic_router import SemanticRouter
from src.router.syntax_router import SyntaxRouter

def calculate_normalized_entropy(probs_dict: dict) -> float:
    """
    Calculates Shannon Entropy and normalizes it to a 0.0 -> 1.0 scale.
    0.0 = Absolute certainty in one class.
    1.0 = Complete confusion across all classes.
    """
    probs = np.array(list(probs_dict.values()))
    probs = np.clip(probs, 1e-9, 1.0) # Prevent log(0)
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(probs))
    
    return float(entropy / max_entropy)

class HybridRouter:
    def __init__(self, t_sem=0.70, t_syn=0.30):
        self.semantic = SemanticRouter()
        self.syntax = SyntaxRouter()
        
        # The Entropy Tripwires
        self.t_sem = t_sem  # Semantic is naturally softer, so it gets a lenient threshold
        self.t_syn = t_syn  # Syntax is naturally sharper, so we demand high certainty

    def load(self, semantic_path="artifacts/semantic_centroids.npy", syntax_path="artifacts/syntax_router.joblib"):
        if not os.path.exists(semantic_path) or not os.path.exists(syntax_path):
            raise FileNotFoundError("Artifacts missing. Run build_semantic_router.py and build_syntax_router.py first.")
            
        print("Loading Semantic and Syntax backends...")
        self.semantic.load(semantic_path)
        self.syntax.load(syntax_path)

    def predict(self, prompt: str, verbose: bool = False, log_file=None):
        """
        Evaluates prompt using Normalized Entropy to measure model confusion.
        Returns: (final_domain, winning_logic_rule)
        """
        sem_probs = self.semantic.predict_proba(prompt)
        syn_probs = self.syntax.predict_proba(prompt)
        
        h_sem = calculate_normalized_entropy(sem_probs)
        h_syn = calculate_normalized_entropy(syn_probs)
        
        sem_top = max(sem_probs, key=sem_probs.get)
        syn_top = max(syn_probs, key=syn_probs.get)

        # ---------------------------------------------------------
        # THE ENTROPY DECISION TREE
        # ---------------------------------------------------------
        if h_sem < self.t_sem:
            decision = sem_top
            rule = f"Semantic (Entropy: {h_sem:.2f} < {self.t_sem})"
        elif h_syn < self.t_syn:
            decision = syn_top
            rule = f"Syntax Override (Entropy: {h_syn:.2f} < {self.t_syn})"
        else:
            decision = "general"
            rule = f"Fallback (Both High Entropy | Sem: {h_sem:.2f}, Syn: {h_syn:.2f})"

        # ---------------------------------------------------------
        # VERBOSE LOGGING
        # ---------------------------------------------------------
        if verbose or log_file:
            clean_prompt = prompt.replace("\n", " ")
            display_prompt = clean_prompt if len(clean_prompt) < 80 else clean_prompt[:77] + "..."
            
            log_output = (
                f"\n[PROMPT] {display_prompt}\n"
                f"  -> SEMANTIC (Top: {sem_top.upper():<8}) | Entropy: {h_sem:.3f}\n"
                f"     Probs: {{'math': {sem_probs.get('math', 0):.3f}, 'code': {sem_probs.get('code', 0):.3f}, 'medical': {sem_probs.get('medical', 0):.3f}, 'general': {sem_probs.get('general', 0):.3f}}}\n"
                f"  -> SYNTAX   (Top: {syn_top.upper():<8}) | Entropy: {h_syn:.3f}\n"
                f"     Probs: {{'math': {syn_probs.get('math', 0):.3f}, 'code': {syn_probs.get('code', 0):.3f}, 'medical': {syn_probs.get('medical', 0):.3f}, 'general': {syn_probs.get('general', 0):.3f}}}\n"
                f"  -> ACTION: Routed to {decision.upper()} via {rule}\n"
            )
            
            if verbose:
                print(log_output, end="")
                
            if log_file:
                # Smart Path Handling
                if log_file is True:
                    final_path = "../../logs/hybrid_logs.txt"
                elif isinstance(log_file, str):
                    final_path = log_file
                    if not final_path.endswith(".txt") and not final_path.endswith(".log"):
                        final_path += ".txt"
                else:
                    final_path = "../../logs/hybrid_logs.txt"

                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                with open(final_path, "a", encoding="utf-8") as f:
                    f.write(log_output)

        return decision, rule