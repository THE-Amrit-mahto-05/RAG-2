import os
import json
import re
import numpy as np
from typing import List, Dict, Optional
from backend.services.embedding_service import EmbeddingService
from backend.api.schema import ImageInfo

class ImageMatcher:
    def __init__(self, image_dir: str = "backend/data/images"):
        self.image_dir = image_dir

    def get_best_image(self, topic_id: str, query_response: str, embedding_service: EmbeddingService, threshold: float = 0.6) -> Optional[ImageInfo]:
        """
        Finds the most relevant image using a hybrid Keyword + Semantic approach.
        """
        topic_image_dir = os.path.join(self.image_dir, topic_id)
        metadata_path = os.path.join(topic_image_dir, "metadata.json")
        embeddings_path = os.path.join(topic_image_dir, "embeddings.npy")
        
        if not os.path.exists(metadata_path) or not os.path.exists(embeddings_path):
            return None
            
        with open(metadata_path, "r") as f:
            images_metadata = json.load(f)
            
        if not images_metadata:
            return None
            
        embeddings = np.load(embeddings_path)
        
        # --- STAGE 1: Keyword-based heuristic Boosting ---
        # Look for explicit keyword matches in descriptions
        query_words = set(re.findall(r'\b\w{4,}\b', query_response.lower()))
        keyword_scores = []
        for meta in images_metadata:
            desc_words = set(re.findall(r'\b\w{4,}\b', meta["description"].lower()))
            overlap = len(query_words.intersection(desc_words))
            keyword_scores.append(overlap / max(len(query_words), 1))
        
        keyword_scores = np.array(keyword_scores)

        # --- STAGE 2: Neural Semantic Search ---
        answer_embedding = embedding_service.get_query_embedding(query_response)
        semantic_scores = np.dot(embeddings, answer_embedding.T).flatten()
        
        # --- STAGE 3: Hybrid Blend & Selection ---
        # We give 60% weight to semantic and 40% to exact keyword matches
        final_scores = (semantic_scores * 0.6) + (keyword_scores * 0.4)
        
        best_idx = np.argmax(final_scores)
        best_score = final_scores[best_idx]
        
        print(f"Image match score: {best_score:.4f} (Semantic: {semantic_scores[best_idx]:.4f}, Keyword: {keyword_scores[best_idx]:.4f})")
        
        # Threshold Check
        if best_score >= threshold:
            best_img = images_metadata[best_idx]
            return ImageInfo(
                url=best_img["url"],
                title=best_img["title"],
                description=best_img["description"]
            )
            
        return None
