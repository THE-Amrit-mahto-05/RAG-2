import fitz
import os
import uuid
import json
from typing import List, Dict, Any
from backend.services.embedding_service import EmbeddingService

class ImageProcessor:
    def __init__(self, image_output_dir: str = "backend/data/images"):
        self.image_output_dir = image_output_dir
        os.makedirs(image_output_dir, exist_ok=True)

    def extract_images(self, pdf_path: str, topic_id: str) -> List[Dict[str, Any]]:
        """Extracts images from the PDF and saves them with metadata."""
        doc = fitz.open(pdf_path)
        topic_image_dir = os.path.join(self.image_output_dir, topic_id)
        os.makedirs(topic_image_dir, exist_ok=True)
        
        extracted_images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                image_id = str(uuid.uuid4())
                image_filename = f"{image_id}.{image_ext}"
                image_path = os.path.join(topic_image_dir, image_filename)
                
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                # In a real scenario, we might use a captioning model here.
                # For this assignment, we use the surrounding text or simple labels.
                extracted_images.append({
                    "id": image_id,
                    "filename": image_filename,
                    "url": f"/api/images/{topic_id}/{image_filename}",
                    "page": page_num + 1,
                    "title": f"Figure from page {page_num + 1}",
                    "description": f"Image extracted from page {page_num + 1} of the textbook."
                })
        
        doc.close()
        
        # Save metadata to disk
        metadata_path = os.path.join(topic_image_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(extracted_images, f)
            
        return extracted_images

    def generate_image_embeddings(self, topic_id: str, extracted_images: List[Dict[str, Any]], embedding_service: EmbeddingService):
        """Generates embeddings for images based on their description/title."""
        topic_image_dir = os.path.join(self.image_output_dir, topic_id)
        
        # We use a combined string of title and description for semantic matching
        texts_to_embed = [f"{img['title']} {img['description']}" for img in extracted_images]
        
        if not texts_to_embed:
            return
            
        embeddings = embedding_service.get_embeddings(texts_to_embed)
        np.save(os.path.join(topic_image_dir, "embeddings.npy"), embeddings)
