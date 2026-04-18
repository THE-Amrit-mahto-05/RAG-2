import os
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, provider: str = "local", model_name: Optional[str] = None):
        """
        Initializes the embedding service.
        Providers: 'local' (Dedicated for lightweight deployment)
        """
        self.provider = "local"
        self.model_name = model_name
        self.dimension = 0
        
        name = model_name or "all-MiniLM-L6-v2"
        print(f"Loading local embedding model: {name}...")
        self.model = SentenceTransformer(name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generates embeddings for a list of strings (supports batching)."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings).astype('float32')
        
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Generates a single normalized embedding for a query."""
        embedding = self.model.encode([query], normalize_embeddings=True)
        return np.array(embedding).astype('float32')
