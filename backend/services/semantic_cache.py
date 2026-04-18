import os
import json
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

class SemanticCache:
    def __init__(self, data_dir: str = "backend/data/cache", max_size: int = 60, threshold: float = 0.95):
        self.data_dir = data_dir
        self.max_size = max_size
        self.threshold = threshold
        os.makedirs(data_dir, exist_ok=True)
        self.cache_file = os.path.join(data_dir, "conversation_cache.json")
        self.embeddings_file = os.path.join(data_dir, "cache_embeddings.npy")
        
        self.cache_entries = [] # list of dicts: {"topic_id": str, "question": str, "response_data": dict, "timestamp": str}
        self.embeddings = None # numpy array of shape (N, D)
        
        self._load()

    def _load(self):
        if os.path.exists(self.cache_file) and os.path.exists(self.embeddings_file):
            try:
                with open(self.cache_file, "r") as f:
                    self.cache_entries = json.load(f)
                self.embeddings = np.load(self.embeddings_file)
                # Trim if somehow larger
                if len(self.cache_entries) > self.max_size:
                    self.cache_entries = self.cache_entries[-self.max_size:]
                    self.embeddings = self.embeddings[-self.max_size:]
                    self._save()
                print(f"Loaded {len(self.cache_entries)} items from Semantic Cache.")
            except Exception as e:
                print(f"Failed to load cache: {e}")
                self.cache_entries = []
                self.embeddings = None

    def _save(self):
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache_entries, f)
            if self.embeddings is not None:
                np.save(self.embeddings_file, self.embeddings)
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def get(self, topic_id: str, question: str, embedding_service) -> Optional[Dict[str, Any]]:
        if not self.cache_entries or self.embeddings is None or len(self.embeddings) == 0:
            return None
            
        try:
            query_embedding = embedding_service.get_query_embedding(question).flatten()
            
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            
            if best_score >= self.threshold:
                match = self.cache_entries[best_idx]
                # Ensure they belong to the same topic!
                if match["topic_id"] == topic_id:
                    print(f"⚡ [CACHE HIT]: Matched '{match['question']}' with {best_score:.2f} confidence.")
                    return match["response_data"]
        except Exception as e:
            print(f"Error reading cache: {e}")
            
        return None

    def put(self, topic_id: str, question: str, response_data: Dict[str, Any], embedding_service):
        try:
            query_embedding = embedding_service.get_query_embedding(question).flatten()
            
            entry = {
                "topic_id": topic_id,
                "question": question,
                "response_data": response_data,
                "timestamp": datetime.now().isoformat()
            }
            
            self.cache_entries.append(entry)
            
            if self.embeddings is None:
                self.embeddings = np.array([query_embedding])
            else:
                self.embeddings = np.vstack([self.embeddings, query_embedding])
                
            # Evict oldest if we exceed max size
            if len(self.cache_entries) > self.max_size:
                self.cache_entries = self.cache_entries[1:]
                self.embeddings = self.embeddings[1:]
                
            self._save()
        except Exception as e:
            print(f"Error updating cache: {e}")
