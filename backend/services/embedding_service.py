from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initializes the embedding model (local)."""
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Dimension: {self.dimension}")

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generates embeddings for a list of strings."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings).astype('float32')

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Generates a single normalized embedding for a query."""
        embedding = self.model.encode([query], normalize_embeddings=True)
        return np.array(embedding).astype('float32')
