import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticRouter:
    def __init__(self, embedding_model="sentence-transformers/all-mpnet-base-v2"):
        self.embedder = SentenceTransformer(embedding_model)
        self.centroids = {}

    def fit(self, domain_data: dict):
        """Calculates centroids for math, code, medical, and general."""
        for domain, prompts in domain_data.items():
            print(f"Embedding {len(prompts)} samples for {domain}...")
            embeddings = self.embedder.encode(prompts, show_progress_bar=True)
            self.centroids[domain] = np.mean(embeddings, axis=0)

    def save(self, filepath="artifacts/semantic_centroids.npy"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, self.centroids)

    def load(self, filepath="artifacts/semantic_centroids.npy"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Centroid file not found at {filepath}")
        self.centroids = np.load(filepath, allow_pickle=True).item()

    def predict_with_score(self, prompt: str):
        if not self.centroids:
            raise ValueError("Router has no centroids. Call fit() or load() first.")

        prompt_embedding = self.embedder.encode([prompt])
        domain_names = list(self.centroids.keys())
        centroid_matrix = np.array(list(self.centroids.values()))

        similarities = cosine_similarity(prompt_embedding, centroid_matrix)[0]
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]

        return domain_names[best_match_idx], best_score

    def predict_proba(self, prompt: str) -> dict:
        """Returns a dictionary of probabilities for all domains."""
        if not self.centroids:
            raise ValueError("Router has no centroids. Call fit() or load() first.")

        prompt_embedding = self.embedder.encode([prompt])
        domain_names = list(self.centroids.keys())
        centroid_matrix = np.array(list(self.centroids.values()))

        similarities = cosine_similarity(prompt_embedding, centroid_matrix)[0]

        # Apply Softmax to mathematically convert cosine distances into a probability distribution
        exp_sims = np.exp(
            similarities - np.max(similarities)
        )  # Subtract max for numerical stability
        probs = exp_sims / exp_sims.sum()

        return {domain_names[i]: float(probs[i]) for i in range(len(domain_names))}

    def predict(self, prompt: str, threshold: float = 0.35):
        """
        Routes the prompt.
        If score < threshold, it defaults to 'general'.
        Returns: (final_decision, best_match_domain, score)
        """
        best_match, score = self.predict_with_score(prompt)

        # The strict gatekeeper: If it's not highly confident, it's general chat.
        if score < threshold:
            return "general", best_match, score

        return best_match, best_match, score
