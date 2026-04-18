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
        self.model_name = model_name or "all-MiniLM-L6-v2"
        self._model = None
        self.dimension = 384 # Known dimension for all-MiniLM-L6-v2

    @property
    def model(self):
        if self._model is None:
            print(f"Lazy loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generates embeddings for a list of strings (supports batching)."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings).astype('float32')
        
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Generates a single normalized embedding for a query."""
        embedding = self.model.encode([query], normalize_embeddings=True)
        return np.array(embedding).astype('float32')
