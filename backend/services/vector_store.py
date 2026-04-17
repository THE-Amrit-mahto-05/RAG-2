import faiss
import os
import json
import numpy as np
from typing import List, Dict, Any
from backend.api.schema import Chunk, Source

class VectorStore:
    def __init__(self, dimension: int, index_path: str = "backend/data/processed"):
        self.dimension = dimension
        self.index_path = index_path
        os.makedirs(index_path, exist_ok=True)

    def create_index(self, topic_id: str, embeddings: np.ndarray, chunks: List[Chunk]):
        """Creates and saves a FAISS index for a specific topic."""
        # Use IndexFlatL2 for small datasets as recommended
        index = faiss.IndexFlatL2(self.dimension)
        index.add(embeddings)
        
        # Save index
        faiss.write_index(index, os.path.join(self.index_path, f"{topic_id}.index"))
        
        # Save chunks as JSON for retrieval
        chunks_json = [chunk.dict() for chunk in chunks]
        with open(os.path.join(self.index_path, f"{topic_id}_chunks.json"), "w") as f:
            json.dump(chunks_json, f)

    def search(self, topic_id: str, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches the FAISS index for the given query embedding."""
        index_file = os.path.join(self.index_path, f"{topic_id}.index")
        chunks_file = os.path.join(self.index_path, f"{topic_id}_chunks.json")
        
        if not os.path.exists(index_file) or not os.path.exists(chunks_file):
            return []

        index = faiss.read_index(index_file)
        with open(chunks_file, "r") as f:
            chunks_data = json.load(f)

        distances, indices = index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(chunks_data):
                chunk_dict = chunks_data[idx]
                # Convert distance to a similarity score (approximate for L2)
                # For normalized vectors, L2 distance d^2 = 2(1 - cos_sim)
                # So cos_sim = 1 - (d^2 / 2)
                similarity = max(0, 1 - (float(dist) / 2))
                results.append({
                    "chunk": Chunk(**chunk_dict),
                    "similarity": similarity
                })
        
        return results
