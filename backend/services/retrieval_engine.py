import re
from typing import List, Dict, Any
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_store import VectorStore
from backend.api.schema import Source, Chunk

class RetrievalEngine:
    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def _extract_page_number(self, query: str) -> List[int]:
        """Detects mentions of page numbers in the query (e.g., 'page 5')."""
        patterns = [
            r'page\s*(\d+)',
            r'p\.\s*(\d+)',
            r'p\s*(\d+)'
        ]
        pages = []
        for pattern in patterns:
            matches = re.findall(pattern, query.lower())
            if matches:
                pages.extend([int(m) for m in matches])
        return list(set(pages))

    def retrieve_context(self, topic_id: str, query: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Retrieves and re-ranks relevant chunks with metadata filtering and diversity."""
        # 1. Detect page filters
        target_pages = self._extract_page_number(query)
        
        # 2. Initial retrieval pool (wider for re-ranking)
        query_embedding = self.embedding_service.get_query_embedding(query)
        initial_results = self.vector_store.search(topic_id, query_embedding, top_k=top_k * 3)
        
        if not initial_results:
            return []

        # 3. Apply Re-ranking and Filtering
        ranked_results = []
        query_terms = set(re.findall(r'\w+', query.lower()))
        
        for res in initial_results:
            chunk = res["chunk"]
            base_score = res["similarity"]
            
            # Simple keyword match boosting
            chunk_text_lower = chunk.text.lower()
            keyword_score = sum(1 for term in query_terms if term in chunk_text_lower) / max(len(query_terms), 1)
            
            # Metadata boosting (Page mentioned)
            page_boost = 1.0
            if target_pages and chunk.page in target_pages:
                page_boost = 1.5 
            
            final_score = (base_score * 0.7) + (keyword_score * 0.3)
            final_score *= page_boost
            
            if final_score >= threshold:
                ranked_results.append({
                    "chunk": chunk,
                    "similarity": final_score,
                    "meta": {
                        "base_score": base_score,
                        "keyword_score": keyword_score,
                        "is_page_match": chunk.page in target_pages if target_pages else False
                    }
                })

        # 4. Sort by final score
        ranked_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # 5. Apply Diversity Control (Limit chunks per page to avoid redundancy)
        diverse_results = []
        page_counts = {}
        for res in ranked_results:
            page = res["chunk"].page
            page_counts[page] = page_counts.get(page, 0) + 1
            
            if page_counts[page] <= 2: # Max 2 chunks per page in top results
                diverse_results.append(res)
            
        # If nothing passed threshold, fall back to the top results anyway
        if not diverse_results and ranked_results:
            diverse_results = sorted(ranked_results, key=lambda x: x["similarity"], reverse=True)[:3]
                
        return diverse_results

    def format_context_for_llm(self, retrieved_results: List[Dict[str, Any]]) -> str:
        """Formats the retrieved chunks into a context string."""
        context_parts = []
        for i, res in enumerate(retrieved_results):
            chunk = res["chunk"]
            context_parts.append(f"[Source {i+1} - Page {chunk.page}]: {chunk.text}")
        
        return "\n---\n".join(context_parts)

    def get_sources_with_text(self, retrieved_results: List[Dict[str, Any]]) -> List[Source]:
        """Converts internal results into API Source objects with full text."""
        return [
            Source(
                chunk_id=res["chunk"].id,
                page=res["chunk"].page,
                similarity=float(res["similarity"]),
                text=res["chunk"].text,
                match_metadata=res.get("meta")
            )
            for res in retrieved_results
        ]
