import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.router.semantic_router import SemanticRouter
from src.router.syntax_router import SyntaxRouter

class MetaRouter:
    def __init__(self):
        self.semantic = SemanticRouter()
        self.syntax = SyntaxRouter()
        self.meta_classifier = RandomForestClassifier(n_estimators=100,max_depth=4,class_weight="balanced",random_state=42)
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
        
        # Build deterministic feature space: [sem_math, sem_code, sem_general, syn_math, syn_code, syn_general]
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
        """
        Trains the meta-gating layer using probability distributions extracted from the sub-routers.
        domain_data: {"math": [...], "code": [...], "hybrid": [...], "general": [...]}
        """
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

        print(f"Training Meta-Classifier on matrix shape {X.shape} (Softmax Optimization)...")
        self.meta_classifier.fit(X, y)
        self.is_meta_trained = True

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.meta_classifier, save_path)
        print(f"Meta-Classifier successfully saved to {save_path}\n")

    def predict(self, prompt: str, verbose: bool = False, log_file=None):
        """Processes runtime queries and directs routing to: math, code, hybrid, or general."""
        if not self.is_meta_trained:
            raise ValueError("Meta-Classifier is not trained or loaded. Call fit() or load() first.")

        # 1. Extract feature array
        X_vec = self._extract_features(prompt)

        # 2. Compute prediction metrics
        probabilities = self.meta_classifier.predict_proba(X_vec)[0]
        classes = self.meta_classifier.classes_
        
        best_idx = np.argmax(probabilities)
        decision = classes[best_idx]
        confidence = probabilities[best_idx]
        
        rule = f"Meta-Classifier Boundary Selection (Confidence: {confidence:.2%})"

        # 3. Format detailed log output to preserve pipeline compatibility
        clean_prompt = prompt.replace("\n", " ")
        display_prompt = clean_prompt if len(clean_prompt) < 80 else clean_prompt[:77] + "..."

        log_output = (
            f"[PROMPT] {display_prompt}\n"
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