from typing import List, Dict, Any
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_store import VectorStore
from backend.api.schema import Source

class RetrievalEngine:
    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve_context(self, topic_id: str, query: str, top_k: int = 5, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Retrieves relevant chunks and filters by similarity threshold."""
        query_embedding = self.embedding_service.get_query_embedding(query)
        search_results = self.vector_store.search(topic_id, query_embedding, top_k=top_k)
        
        # Filter by threshold and format
        filtered_results = []
        for res in search_results:
            if res["similarity"] >= threshold:
                filtered_results.append(res)
        
        return filtered_results

    def format_context_for_llm(self, retrieved_results: List[Dict[str, Any]]) -> str:
        """Formats the retrieved chunks into a context string for the LLM."""
        context_parts = []
        for i, res in enumerate(retrieved_results):
            chunk = res["chunk"]
            context_parts.append(f"[Chunk {i+1} - Page {chunk.page}]: {chunk.text}")
        
        return "\n---\n".join(context_parts)
