import faiss
import os
import json
import numpy as np
from typing import List, Dict, Any
from backend.api.schema import Chunk

class VectorStore:
    def __init__(self, dimension: int, base_path: str = "backend/data/topics"):
        self.dimension = dimension
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def _get_topic_dir(self, topic_id: str) -> str:
        topic_dir = os.path.join(self.base_path, topic_id)
        os.makedirs(topic_dir, exist_ok=True)
        return topic_dir

    def create_index(self, topic_id: str, embeddings: np.ndarray, chunks: List[Chunk], metadata: Dict[str, Any] = {}):
        """Creates and saves a FAISS index and associated data for a specific topic."""
        topic_dir = self._get_topic_dir(topic_id)
        
        # 1. faiss_index.bin
        index = faiss.IndexFlatL2(self.dimension)
        index.add(embeddings)
        faiss.write_index(index, os.path.join(topic_dir, "faiss_index.bin"))
        
        # 2. embeddings.npy
        np.save(os.path.join(topic_dir, "embeddings.npy"), embeddings)
        
        # 3. chunks.json
        chunks_json = [chunk.dict() for chunk in chunks]
        with open(os.path.join(topic_dir, "chunks.json"), "w") as f:
            json.dump(chunks_json, f)
            
        # 4. metadata.json
        full_metadata = {
            "topic_id": topic_id,
            "dimension": self.dimension,
            "chunk_count": len(chunks),
            **metadata
        }
        with open(os.path.join(topic_dir, "metadata.json"), "w") as f:
            json.dump(full_metadata, f)

    def search(self, topic_id: str, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches the topic's FAISS index."""
        topic_dir = os.path.join(self.base_path, topic_id)
        index_file = os.path.join(topic_dir, "faiss_index.bin")
        chunks_file = os.path.join(topic_dir, "chunks.json")
        
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
                similarity = max(0, 1 - (float(dist) / 2))
                results.append({
                    "chunk": Chunk(**chunk_dict),
                    "similarity": similarity
                })
        
        return results
