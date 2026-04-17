import os
import json
import re
import numpy as np
from typing import List, Dict, Optional
from backend.services.embedding_service import EmbeddingService
from backend.api.schema import ImageInfo

class ImageMatcher:
    def __init__(self, static_image_dir: str = None):
        if static_image_dir is None:
            # Dynamically resolve backend/data/Sound using __file__
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.static_image_dir = os.path.join(base_dir, "data", "Sound")
        else:
            self.static_image_dir = static_image_dir

    def get_best_image(self, topic_id: str, query_response: str, embedding_service: EmbeddingService, threshold: float = 0.4) -> Optional[ImageInfo]:
        """
        Finds the most relevant image using a hybrid Keyword + Semantic approach.
        """
        # Hardcoded to use the static assignment images directory
        metadata_path = os.path.join(self.static_image_dir, "images.json")
        embeddings_path = os.path.join(self.static_image_dir, "embeddings.npy")
        
        if not os.path.exists(metadata_path) or not os.path.exists(embeddings_path):
            return None
            
        with open(metadata_path, "r") as f:
            images_metadata = json.load(f)
            
        # --- EVALUATION REQUIREMENT: Explicit Text Matching ---
        # "If the answer mentions 'vibration' or 'longitudinal', return the corresponding image URL."
        lower_query = query_response.lower()
        print("DEBUG: ImageMatcher hit! lower_query length:", len(lower_query))
        print("DEBUG: first 50 chars:", lower_query[:50])
        
        # Mapping explicit keywords to their required static diagrams
        keyword_map = {
            "reflection": "ReflectionOfSound.png",
            "longitudinal": "CompressionAndRefraction.png",
            "compression": "CompressionAndRefraction.png",
            "rubber": "VibrationOfRubberBand.png",
            "bell": "SchoolBellVibration.png",
            "instrument": "MusicalInstrumentsVibrationChart.png",
            "vocal": "VocalCordsDiagram.png",
            "vibration": "SchoolBellVibration.png" # Secondary fallback
        }
        
        # 1. Exact string matching (Highest Priority)
        for keyword, filename in keyword_map.items():
            if keyword in lower_query:
                print(f"DEBUG: Found keyword '{keyword}' mapped to '{filename}'")
                # Find corresponding metadata
                for meta in images_metadata:
                    if filename in meta["path"]:
                        print(f"DEBUG: Found matching metadata for {filename}: {meta['title']}")
                        return ImageInfo(
                            url=meta["url"],
                            title=meta["title"],
                            description=meta["description"]
                        )
                print(f"DEBUG: ERROR - Keyword '{keyword}' matched but '{filename}' was NOT found in images_metadata paths!")

        # 2. Semantic Fallback
        embeddings = np.load(embeddings_path)
        answer_embedding = embedding_service.get_query_embedding(query_response)
        semantic_scores = np.dot(embeddings, answer_embedding.T).flatten()
        
        best_idx = np.argmax(semantic_scores)
        if semantic_scores[best_idx] >= 0.2:
            best_img = images_metadata[best_idx]
            return ImageInfo(
                url=best_img["url"],
                title=best_img["title"],
                description=best_img["description"]
            )
            
        return None
