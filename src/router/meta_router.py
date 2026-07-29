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
            hybrid_margin: float = 0.0,         # Dropped to 0 (Trust the RF's margin)
            hybrid_min_confidence: float = 0.30,
            hybrid_min_existence: float = 0.0   # Dropped to 0 (Stop vetoing equations)
        ):
            if not self.is_meta_trained:
                raise ValueError("Meta-Classifier is not trained or loaded. Call fit() or load() first.")

            sem_probs = self.semantic.predict_proba(prompt)
            syn_probs = self.syntax.predict_proba(prompt)
            
            sem_math = sem_probs.get("math", 0.0)
            sem_code = sem_probs.get("code", 0.0)
            sem_gen  = sem_probs.get("general", 0.0)
            
            X_vec = np.array([[
                sem_math, sem_code, sem_gen,
                syn_probs.get("math", 0.0),
                syn_probs.get("code", 0.0),
                syn_probs.get("general", 0.0)
            ]])

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

            # ==========================================
            # 3. ANTI-HALLUCINATION GUARDRAILS
            # ==========================================
            
            # Guard A: The General Syntax Hallucination Check
            # If the RF chooses General purely based on syntax, but the semantic engine 
            # recognizes dense technical content (SemGen < 0.28), override to the fallback.
            if decision == "general" and sem_gen < 0.28:
                decision = fallback_decision
                rule = f"Semantic Veto of GENERAL (SemGen {sem_gen:.3f} < 0.28) -> Fallback to {decision.upper()}"
                
            # Guard B: Base Hybrid Weak Confidence
            elif decision == "hybrid" and confidence < hybrid_min_confidence:
                decision = fallback_decision
                rule = f"Hybrid Weak Confidence ({confidence:.1%} < {hybrid_min_confidence:.0%}) -> Fallback to {decision.upper()}"

            # 4. Format detailed log output
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