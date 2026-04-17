import os
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer

# Optional imports for cloud providers
try:
    import openai
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

class EmbeddingService:
    def __init__(self, provider: str = "local", model_name: Optional[str] = None):
        """
        Initializes the embedding service.
        Providers: 'local', 'openai', 'gemini'
        """
        self.provider = provider
        self.model_name = model_name
        self.dimension = 0
        
        if provider == "local":
            name = model_name or "all-MiniLM-L6-v2"
            print(f"Loading local embedding model: {name}...")
            self.model = SentenceTransformer(name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        elif provider == "openai":
            if not OpenAI:
                raise ImportError("OpenAI package not installed. Run 'pip install openai'")
            name = model_name or "text-embedding-3-small"
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.dimension = 1536 # Default for text-embedding-3-small
            print(f"Initialized OpenAI embedding: {name}")
        elif provider == "gemini":
            if not genai:
                raise ImportError("Google Generative AI package not installed. Run 'pip install google-generativeai'")
            name = model_name or "models/embedding-001"
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.dimension = 768 # Default for models/embedding-001
            print(f"Initialized Gemini embedding: {name}")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generates embeddings for a list of strings (supports batching)."""
        if self.provider == "local":
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            return np.array(embeddings).astype('float32')
        
        elif self.provider == "openai":
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name or "text-embedding-3-small"
            )
            embeddings = [data.embedding for data in response.data]
            return np.array(embeddings).astype('float32')
            
        elif self.provider == "gemini":
            # Gemini typically handles batching well
            result = genai.embed_content(
                model=self.model_name or "models/embedding-001",
                content=texts,
                task_type="retrieval_document"
            )
            return np.array(result['embedding']).astype('float32')

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Generates a single normalized embedding for a query."""
        if self.provider == "local":
            embedding = self.model.encode([query], normalize_embeddings=True)
            return np.array(embedding).astype('float32')
            
        elif self.provider == "openai":
            response = self.client.embeddings.create(
                input=[query],
                model=self.model_name or "text-embedding-3-small"
            )
            return np.array([response.data[0].embedding]).astype('float32')
            
        elif self.provider == "gemini":
            result = genai.embed_content(
                model=self.model_name or "models/embedding-001",
                content=query,
                task_type="retrieval_query"
            )
            return np.array([result['embedding']]).astype('float32')
