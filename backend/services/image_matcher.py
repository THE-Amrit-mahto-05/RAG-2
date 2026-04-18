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

    def get_best_image(self, topic_id: str, query_response: str, embedding_service: EmbeddingService, threshold: float = 0.4, context_pages: list = None) -> Optional[ImageInfo]:
        """Returns the single best matching image (legacy method)."""
        results = self.get_top_images(topic_id, query_response, embedding_service, threshold=threshold, context_pages=context_pages, top_n=1)
        return results[0] if results else None

    def get_top_images(self, topic_id: str, query_response: str, embedding_service: EmbeddingService, threshold: float = 0.45, context_pages: list = None, top_n: int = 3) -> list:
        """
        Returns top-N most relevant images ranked by semantic similarity + page-context boost.
        Strictly filters out poor matches to avoid irrelevant figures.
        """
        topic_dir = os.path.join(self.data_dir, topic_id)
        metadata_path = os.path.join(topic_dir, "metadata.json")
        embeddings_path = os.path.join(topic_dir, "embeddings.npy")
        
        if not os.path.exists(metadata_path) or not os.path.exists(embeddings_path):
            return []
            
        with open(metadata_path, "r") as f:
            images_metadata = json.load(f)
            
        try:
            image_embeddings = np.load(embeddings_path)
            
            # Keyword overlap scoring
            words_in_query = set(re.findall(r'\b\w+\b', query_response.lower()))
            keyword_scores = []
            for meta in images_metadata:
                desc_words = set(re.findall(r'\b\w+\b', meta["description"].lower()))
                keyword_scores.append(len(words_in_query.intersection(desc_words)))

            # Semantic similarity
            query_embedding = embedding_service.get_query_embedding(query_response).flatten()
            similarities = np.dot(image_embeddings, query_embedding) / (
                np.linalg.norm(image_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
            )
            
            # Apply keyword boost
            for idx, kw_score in enumerate(keyword_scores):
                similarities[idx] += kw_score * 0.10

            # Apply page-context boost (strongest signal)
            if context_pages:
                for idx, meta in enumerate(images_metadata):
                    if meta.get("page") in context_pages:
                        similarities[idx] += 0.35

            # Rank all images
            ranked_indices = np.argsort(similarities)[::-1]
            
            results = []
            seen_pages = set()
            for idx in ranked_indices:
                if len(results) >= top_n:
                    break
                score = float(similarities[idx])
                if score < threshold:
                    break
                meta = images_metadata[idx]
                # Avoid showing 2 images from the exact same page
                if meta.get("page") in seen_pages:
                    continue
                seen_pages.add(meta.get("page"))
                results.append(ImageInfo(
                    url=meta["url"],
                    title=meta["title"],
                    description=meta["description"]
                ))
            
            return results
            
        except Exception as e:
            print(f"Error during image matching: {e}")
            return []
