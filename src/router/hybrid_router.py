import os
import numpy as np
from src.router.semantic_router import SemanticRouter
from src.router.syntax_router import SyntaxRouter

class HybridRouter:
    def __init__(self, t_sem=0.25, t_syn=0.50):
        self.semantic = SemanticRouter()
        self.syntax = SyntaxRouter()
        self.t_sem = t_sem
        self.t_syn = t_syn

    def load(self, semantic_path="models/semantic_centroids.npy", syntax_path="models/syntax_router.joblib"):
        if not os.path.exists(semantic_path) or not os.path.exists(syntax_path):
            raise FileNotFoundError("Models missing. Check the models/ directory.")
        self.semantic.load(semantic_path)
        self.syntax.load(syntax_path)

    def predict(self, prompt: str, verbose: bool = False, log_file=None):
        sem_probs = self.semantic.predict_proba(prompt)
        syn_probs = self.syntax.predict_proba(prompt)
        
        sem_top = max(sem_probs, key=sem_probs.get)
        sem_score = sem_probs[sem_top]
        
        syn_top = max(syn_probs, key=syn_probs.get)
        syn_score = syn_probs[syn_top]

        sem_valid = sem_score >= self.t_sem
        syn_valid = syn_score >= self.t_syn

        # ---------------------------------------------------------
        # DECISION MATRIX (With Clean Rule Names for Aggregation)
        # ---------------------------------------------------------
        if not sem_valid and not syn_valid:
            decision = "general"
            rule = "Fallback (Both failed)"
            
        elif sem_valid and not syn_valid:
            decision = sem_top
            rule = "Semantic Solo Win"
            
        elif syn_valid and not sem_valid:
            decision = syn_top
            rule = "Syntax Solo Win"
            
        else:
            if sem_top == syn_top:
                decision = sem_top
                rule = "Strong Agreement"
                
            elif sem_top in ["math", "medical"]:
                decision = sem_top
                rule = "Semantic Domain Authority"
                
            elif syn_top == "code":
                decision = syn_top
                rule = "Syntax Code Authority"
                
            elif sem_top == "code" and syn_top == "general":
                # THE FIX: Stops Semantic from dragging General prompts into Code
                decision = syn_top
                rule = "Syntax General Authority"
                
            else:
                decision = sem_top
                rule = "Semantic Default Authority"

        # ---------------------------------------------------------
        # FORMAT DETAILED LOG STRING
        # ---------------------------------------------------------
        clean_prompt = prompt.replace("\n", " ")
        display_prompt = clean_prompt if len(clean_prompt) < 80 else clean_prompt[:77] + "..."
        
        log_output = (
            f"[PROMPT] {display_prompt}\n"
            f"  -> SEMANTIC | Top: {sem_top:<8} | Score: {sem_score:.3f} | Valid: {sem_valid}\n"
            f"  -> SYNTAX   | Top: {syn_top:<8} | Score: {syn_score:.3f} | Valid: {syn_valid}\n"
            f"  -> ACTION: Routed to {decision.upper()} via {rule}\n"
        )
        
        if verbose:
            print(f"\n{log_output}", end="")
            
        if log_file:
            final_path = log_file if isinstance(log_file, str) else "logs/hybrid_logs.txt"
            if not final_path.endswith(".txt") and not final_path.endswith(".log"): final_path += ".txt"
            if "/" not in final_path and "\\" not in final_path: final_path = os.path.join("logs", final_path)
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            with open(final_path, "a", encoding="utf-8") as f:
                f.write(f"\n{log_output}")

        # Return the log string so the eval script can use it for errors
        return decision, rule, log_output