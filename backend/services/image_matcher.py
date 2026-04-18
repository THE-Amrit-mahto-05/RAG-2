import os
import json
import re
import numpy as np
from typing import Optional
from services.embedding_service import EmbeddingService
from api.schema import ImageInfo

class ImageMatcher:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Dynamically set base to backend/data/topics
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(base_dir, "data", "topics")
        else:
            self.data_dir = data_dir

    def get_best_image(self, topic_id: str, query_response: str, embedding_service: EmbeddingService, threshold: float = 0.4) -> Optional[ImageInfo]:
        """
        Dynamically finds the most relevant image using a Semantic approach on the specific uploaded topic's Images.
        """
        topic_dir = os.path.join(self.data_dir, topic_id)
        metadata_path = os.path.join(topic_dir, "metadata.json")
        embeddings_path = os.path.join(topic_dir, "embeddings.npy")
        
        # If the generated PDF directory does not have manually associated images, gracefully return
        if not os.path.exists(metadata_path) or not os.path.exists(embeddings_path):
            return None
            
        with open(metadata_path, "r") as f:
            images_metadata = json.load(f)
            
        try:
            image_embeddings = np.load(embeddings_path)
            
            # --- Keyword Boost Logic ---
            # If the query contains discrete keywords, check for direct occurrences in descriptions
            words_in_query = set(re.findall(r'\b\w+\b', query_response.lower()))
            keyword_match_idx = -1
            max_kw_overlap = 0

            for idx, meta in enumerate(images_metadata):
                desc_words = set(re.findall(r'\b\w+\b', meta["description"].lower()))
                overlap = len(words_in_query.intersection(desc_words))
                if overlap > max_kw_overlap:
                    max_kw_overlap = overlap
                    keyword_match_idx = idx

            # Semantic search
            query_embedding = embedding_service.get_query_embedding(query_response).flatten()
            
            similarities = np.dot(image_embeddings, query_embedding) / (
                np.linalg.norm(image_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Boost the score if we found a strong keyword overlap
            if keyword_match_idx != -1:
                similarities[keyword_match_idx] += (max_kw_overlap * 0.15) 

            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            
            if best_score > threshold:
                best_meta = images_metadata[best_idx]
                return ImageInfo(
                    url=best_meta["url"],
                    title=best_meta["title"],
                    description=best_meta["description"]
                )
        except Exception as e:
            print(f"Error during semantic image matching: {e}")
            
        return None
