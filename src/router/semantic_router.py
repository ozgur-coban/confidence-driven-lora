import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticRouter:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        """Initializes the router with a lightweight embedding model."""
        self.embedder = SentenceTransformer(embedding_model)
        self.centroids = {}

    def fit(self, domain_data: dict):
        """
        Calculates the centroids for each domain.
        domain_data format: {"math": [prompt1, prompt2], "code": [...]}
        """
        for domain, prompts in domain_data.items():
            print(f"Embedding {len(prompts)} samples for {domain}...")
            embeddings = self.embedder.encode(prompts, show_progress_bar=True)
            self.centroids[domain] = np.mean(embeddings, axis=0)
            
    def save(self, filepath="artifacts/router_centroids.npy"):
        """Saves the computed centroids to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, self.centroids)
        
    def load(self, filepath="artifacts/router_centroids.npy"):
        """Loads pre-computed centroids for fast inference."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Centroid file not found at {filepath}")
        
        # item() extracts the dictionary from the 0-d numpy array
        self.centroids = np.load(filepath, allow_pickle=True).item()
        
    def predict_with_score(self, prompt: str):
            """Returns the predicted domain and its confidence score."""
            if not self.centroids:
                raise ValueError("Router has no centroids. Call fit() or load() first.")
                
            prompt_embedding = self.embedder.encode([prompt])
            domain_names = list(self.centroids.keys())
            centroid_matrix = np.array(list(self.centroids.values()))
            
            similarities = cosine_similarity(prompt_embedding, centroid_matrix)[0]
            best_match_idx = np.argmax(similarities)
            best_score = similarities[best_match_idx]
            
            return domain_names[best_match_idx], best_score
            
    def predict(self, prompt: str, threshold: float = 0.45) -> str:
            domain, score = self.predict_with_score(prompt)
            if score < threshold:
                return "general"
            return domain