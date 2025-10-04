import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class Recommender:
    def __init__(self, csv_path: str, embeddings_path: str):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)
        if not os.path.isfile(embeddings_path):
            raise FileNotFoundError(embeddings_path)
        self.products = pd.read_csv(csv_path)
        self.embeddings = np.load(embeddings_path)
        if len(self.products) != len(self.embeddings):
            raise ValueError("Products and embeddings count mismatch")

    def recommend_by_index(self, index: int, top_k: int = 5):
        if index < 0 or index >= len(self.products):
            raise IndexError("index out of range")
        query_vec = self.embeddings[index:index+1]
        sims = cosine_similarity(query_vec, self.embeddings).flatten()
        # exclude self
        sims[index] = -1.0
        top_idx = np.argsort(-sims)[:top_k]
        return self.products.iloc[top_idx].assign(score=sims[top_idx]).reset_index(drop=True)


