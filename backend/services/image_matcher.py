import os
import json
import numpy as np
from typing import List, Dict, Optional
from backend.services.embedding_service import EmbeddingService
from backend.api.schema import ImageInfo

class ImageMatcher:
    def __init__(self, image_dir: str = "backend/data/images"):
        self.image_dir = image_dir

    def get_best_image(self, topic_id: str, query_response: str, embedding_service: EmbeddingService, threshold: float = 0.5) -> Optional[ImageInfo]:
        """Finds the most relevant image for a given LLM response."""
        topic_image_dir = os.path.join(self.image_dir, topic_id)
        metadata_path = os.path.join(topic_image_dir, "metadata.json")
        embeddings_path = os.path.join(topic_image_dir, "embeddings.npy")
        
        if not os.path.exists(metadata_path) or not os.path.exists(embeddings_path):
            return None
            
        with open(metadata_path, "r") as f:
            images_metadata = json.load(f)
            
        embeddings = np.load(embeddings_path)
        
        if len(images_metadata) == 0:
            return None
            
        # Strategy: Semantic Similarity between LLM's Answer and Image Metadata
        # We could also use keywords extracted from the answer, but semantic is more robust.
        answer_embedding = embedding_service.get_query_embedding(query_response)
        
        # Calculate cosine similarity: (A . B) / (||A|| * ||B||)
        # Since we use normalized embeddings, it's just the dot product
        similarities = np.dot(embeddings, answer_embedding.T).flatten()
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score >= threshold:
            best_img = images_metadata[best_idx]
            return ImageInfo(
                url=best_img["url"],
                title=best_img["title"],
                description=best_img["description"]
            )
            
        return None
