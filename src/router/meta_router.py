import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.router.semantic_router import SemanticRouter
from src.router.syntax_router import SyntaxRouter

class MetaRouter:
    FEATURE_NAMES = [
        "sem_math", "sem_code", "sem_general",
        "syn_math", "syn_code", "syn_general"
    ]

    def __init__(self):
        self.semantic = SemanticRouter()
        self.syntax = SyntaxRouter()
        self.meta_classifier = RandomForestClassifier(
            n_estimators=100, max_depth=4, class_weight="balanced", random_state=42
        )
        self.is_meta_trained = False

    def load(
        self,
        semantic_path="models/semantic_centroids.npy",
        syntax_path="models/syntax_router.joblib",
        meta_path="models/meta_router.joblib",
    ):
        """Loads all sub-routers and attempts to load the meta-classifier if it exists."""
        if not os.path.exists(semantic_path) or not os.path.exists(syntax_path):
            raise FileNotFoundError("Sub-model artifacts missing. Check models/ directory.")
        
        self.semantic.load(semantic_path)
        self.syntax.load(syntax_path)
        print(" -> Base Semantic and Syntax routers loaded successfully.")

        if os.path.exists(meta_path):
            self.meta_classifier = joblib.load(meta_path)
            self.is_meta_trained = True
            print(f" -> Meta-Classifier loaded successfully from {meta_path}")
        else:
            print(" -> Warning: No trained Meta-Classifier found. You must run fit() before inferencing.")

    def _extract_features(self, prompt: str) -> np.ndarray:
        """Helper to pass text through base routers and extract the 6-dimensional probability vector."""
        sem_probs = self.semantic.predict_proba(prompt)
        syn_probs = self.syntax.predict_proba(prompt)
        
        features = [
            sem_probs.get("math", 0.0),
            sem_probs.get("code", 0.0),
            sem_probs.get("general", 0.0),
            syn_probs.get("math", 0.0),
            syn_probs.get("code", 0.0),
            syn_probs.get("general", 0.0)
        ]
        return np.array([features])

    def fit(self, domain_data: dict, save_path="models/meta_router.joblib"):
        """Trains the meta-gating layer and outputs feature importance analysis."""
        X_features = []
        y_labels = []

        print("\nExtracting meta-features from training prompts...")
        for domain, prompts in domain_data.items():
            print(f" -> Processing features for domain: '{domain}' ({len(prompts)} samples)")
            for prompt in prompts:
                feat = self._extract_features(prompt)[0]
                X_features.append(feat)
                y_labels.append(domain)

        X = np.array(X_features)
        y = np.array(y_labels)

        print(f"Training Meta-Classifier on matrix shape {X.shape}...")
        self.meta_classifier.fit(X, y)
        self.is_meta_trained = True

        # --- DIAGNOSTIC: FEATURE IMPORTANCE PRINT ---
        print("\n" + "=" * 50)
        print("RANDOM FOREST FEATURE IMPORTANCES")
        print("=" * 50)
        importances = self.meta_classifier.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        for idx in sorted_indices:
            print(f" - {self.FEATURE_NAMES[idx]:<12}: {importances[idx]:.4f} ({importances[idx]*100:.1f}%)")
        print("=" * 50 + "\n")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.meta_classifier, save_path)
        print(f"Meta-Classifier successfully saved to {save_path}\n")

    def predict(
            self, 
            prompt: str, 
            verbose: bool = False, 
            log_file=None, 
            hybrid_margin: float = 0.15,
            hybrid_min_confidence: float = 0.50,
            hybrid_min_existence: float = 0.05
        ):
            if not self.is_meta_trained:
                raise ValueError("Meta-Classifier is not trained or loaded. Call fit() or load() first.")

            # 1. Base probabilities
            sem_probs = self.semantic.predict_proba(prompt)
            syn_probs = self.syntax.predict_proba(prompt)
            
            X_vec = np.array([[
                sem_probs.get("math", 0.0),
                sem_probs.get("code", 0.0),
                sem_probs.get("general", 0.0),
                syn_probs.get("math", 0.0),
                syn_probs.get("code", 0.0),
                syn_probs.get("general", 0.0)
            ]])

            # 2. Compute Meta-Classifier predictions
            probabilities = self.meta_classifier.predict_proba(X_vec)[0]
            classes = self.meta_classifier.classes_
            
            sorted_indices = np.argsort(probabilities)[::-1]
            best_idx = sorted_indices[0]
            fallback_idx = sorted_indices[1]
            
            decision = classes[best_idx]
            confidence = probabilities[best_idx]
            
            fallback_decision = classes[fallback_idx]
            fallback_confidence = probabilities[fallback_idx]

            rule = f"Meta-Classifier Boundary Selection (Confidence: {confidence:.2%})"

            # --- 3. HYBRID STRICTNESS GUARDRAILS ---
            if decision == "hybrid":
                delta = confidence - fallback_confidence
                
                math_missing = syn_probs.get("math", 0.0) < 0.01
                code_missing = syn_probs.get("code", 0.0) < hybrid_min_existence
                
                proposed_decision = decision
                proposed_rule = rule
                
                if math_missing or code_missing:
                    for idx in sorted_indices:
                        candidate = classes[idx]
                        if candidate == "hybrid":
                            continue
                        if candidate == "math" and math_missing:
                            continue
                        if candidate == "code" and code_missing:
                            continue
                        
                        proposed_decision = candidate
                        if math_missing and not code_missing:
                            proposed_rule = f"Missing Ingredient Veto (Math Syntax {syn_probs.get('math', 0.0):.4f} < 0.01) -> Fallback to {candidate.upper()}"
                        elif code_missing and not math_missing:
                            proposed_rule = f"Missing Ingredient Veto (Code Syntax {syn_probs.get('code', 0.0):.4f} < {hybrid_min_existence}) -> Fallback to {candidate.upper()}"
                        else:
                            proposed_rule = f"Missing Ingredient Veto (Math & Code Syntax Missing) -> Fallback to {candidate.upper()}"
                        break
                        
                elif confidence < hybrid_min_confidence:
                    proposed_decision = fallback_decision
                    proposed_rule = (
                        f"Hybrid Weak Confidence ({confidence:.1%} < {hybrid_min_confidence:.0%}) "
                        f"-> Fallback to {fallback_decision.upper()}"
                    )
                
                elif delta < hybrid_margin:
                    proposed_decision = fallback_decision
                    proposed_rule = (
                        f"Hybrid Margin Downgrade (Delta {delta:.3f} < {hybrid_margin}) "
                        f"-> Fallback to {fallback_decision.upper()}"
                    )

                if proposed_decision == "general":
                    decision = classes[best_idx]
                    rule = f"Guardrail Abort (Prevented drop to GENERAL) -> Reverted to {decision.upper()} (Confidence: {confidence:.2%})"
                else:
                    decision = proposed_decision
                    rule = proposed_rule

            # 4. Format detailed log output (includes raw feature vector)
            clean_prompt = prompt.replace("\n", " ")
            display_prompt = clean_prompt if len(clean_prompt) < 80 else clean_prompt[:77] + "..."

            log_output = (
                f"[PROMPT] {display_prompt}\n"
                f"  -> INPUT VECTOR (6D): "
                f"[SemMath: {X_vec[0][0]:.3f}, SemCode: {X_vec[0][1]:.3f}, SemGen: {X_vec[0][2]:.3f} | "
                f"SynMath: {X_vec[0][3]:.3f}, SynCode: {X_vec[0][4]:.3f}, SynGen: {X_vec[0][5]:.3f}]\n"
                f"  -> META DISTRIBUTION:\n"
                + "".join([f"     - {classes[i]:<8}: {probabilities[i]:.4f}\n" for i in range(len(classes))]) +
                f"  -> ACTION: Routed to {decision.upper()} via {rule}\n"
            )

            if verbose:
                print(f"\n{log_output}", end="")

            if log_file:
                final_path = log_file if isinstance(log_file, str) else "logs/meta_router_logs.txt"
                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                with open(final_path, "a", encoding="utf-8") as f:
                    f.write(f"\n{log_output}")

            return decision, rule, log_output