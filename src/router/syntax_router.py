import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class SyntaxRouter:
    def __init__(self):

        self.vectorizer = TfidfVectorizer(
            analyzer='word', 
            token_pattern=r'(?u)\b\w+\b|[^\w\s]', 
            stop_words='english',
            ngram_range=(1, 3), 
            min_df=100, 
            max_df=0.5,
            max_features=10000
        )
        

        self.classifier = LogisticRegression(max_iter=1000, class_weight=None)
        self.is_fitted = False

    def fit(self, domain_data: dict):
        """
        Trains the router.
        domain_data: {"math": [list of strings], "code": [list of strings], ...}
        """
        X_text = []
        y_labels = []
        for domain, prompts in domain_data.items():
            X_text.extend(prompts)
            y_labels.extend([domain] * len(prompts))
            
        print("Vectorizing character n-grams...")
        X_vectorized = self.vectorizer.fit_transform(X_text)
        
        print("Training Logistic Regression classifier (Softmax)...")
        self.classifier.fit(X_vectorized, y_labels)
        self.is_fitted = True
        
    def get_top_identifiers(self, n: int = 15) -> dict:
        """
        Extracts the highest-weighted character sequences for each domain.
        This effectively tells you 'what the model thinks Code looks like'.
        """
        if not self.is_fitted:
            raise ValueError("Router must be fitted before extracting identifiers.")
            
        feature_names = self.vectorizer.get_feature_names_out()
        classes = self.classifier.classes_
        top_identifiers = {}
        
        for i, class_label in enumerate(classes):
            # Get the coefficients for this specific class
            # (Higher value = stronger signal for this domain)
            coefs = self.classifier.coef_[i]
            
            # Get indices of the top N largest coefficients
            top_indices = np.argsort(coefs)[-n:]
            
            # Map indices back to the actual n-gram strings
            # We reverse it so the highest weighted is first
            top_features = [feature_names[j] for j in reversed(top_indices)]
            top_identifiers[class_label] = top_features
            
        return top_identifiers

    def save(self, filepath="artifacts/syntax_router.joblib"):
        """Saves the entire pipeline (vectorizer + classifier) to a single file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({'vectorizer': self.vectorizer, 'classifier': self.classifier}, filepath)
        print(f"Router saved to {filepath}")
        
    def load(self, filepath="artifacts/syntax_router.joblib"):
        """Loads a pre-trained router."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Router file not found at {filepath}")
        
        data = joblib.load(filepath)
        self.vectorizer = data['vectorizer']
        self.classifier = data['classifier']
        self.is_fitted = True

    def predict_with_score(self, prompt: str):
        """Returns the predicted domain and its Softmax probability (0.0 to 1.0)."""
        if not self.is_fitted:
            raise ValueError("Router has not been fitted or loaded.")
            
        # Transform the single prompt into the same n-gram vector space
        X_vec = self.vectorizer.transform([prompt])
        
        # predict_proba returns array like [[0.1, 0.8, 0.1]]
        probabilities = self.classifier.predict_proba(X_vec)[0]
        
        best_match_idx = np.argmax(probabilities)
        best_score = probabilities[best_match_idx]
        predicted_domain = self.classifier.classes_[best_match_idx]
        
        return predicted_domain, best_score
    def predict_proba(self, prompt: str) -> dict:
        """Returns a dictionary of probabilities for all domains."""
        if not self.is_fitted:
            raise ValueError("Router has not been fitted or loaded.")
            
        X_vec = self.vectorizer.transform([prompt])
        probs = self.classifier.predict_proba(X_vec)[0]
        classes = self.classifier.classes_
        
        return {classes[i]: float(probs[i]) for i in range(len(classes))}
        
    def predict(self, prompt: str, threshold: float = 0.50) -> str:
        """
        Routes the prompt. 
        If confidence is below threshold, returns 'general' (the fallback).
        """
        domain, score = self.predict_with_score(prompt)
        if score < threshold:
            return "general"
        return domain