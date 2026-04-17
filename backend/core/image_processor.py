import os
import json
import fitz # PyMuPDF
import numpy as np
from typing import List, Dict, Any
from backend.services.embedding_service import EmbeddingService

class ImageProcessor:
    def __init__(self, base_image_dir: str = "backend/data/images"):
        self.base_image_dir = base_image_dir
        os.makedirs(base_image_dir, exist_ok=True)

    def extract_images(self, pdf_path: str, topic_id: str) -> List[Dict[str, Any]]:
        """Extracts significant images from PDF and attempts to find matching captions."""
        doc = fitz.open(pdf_path)
        topic_dir = os.path.join(self.base_image_dir, topic_id)
        os.makedirs(topic_dir, exist_ok=True)
        
        extracted_images = []
        image_count = 0

        for i in range(len(doc)):
            page = doc[i]
            # Try to get text blocks near images for caption matching
            blocks = page.get_text("blocks")
            
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Filter by resolution (Ignore tiny icons/bullets)
                if base_image["width"] < 150 or base_image["height"] < 150:
                    continue

                image_ext = base_image["ext"]
                image_filename = f"image_{image_count}.{image_ext}"
                image_path = os.path.join(topic_dir, image_filename)
                
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                # HEURISTIC: Look for 'Figure X' or 'Diagram' near the image
                # In PyMuPDF, images have a 'rect' on the page. We check text blocks near that rect.
                caption = f"Figure from page {i+1}"
                img_bbox = page.get_image_bbox(img) # Get image rect
                
                # Check blocks directly above or below the image rect
                for b in blocks:
                    block_text = b[4].strip()
                    # If block is nearby and contains 'Figure' or 'Fig'
                    if (abs(b[1] - img_bbox[3]) < 50 or abs(b[3] - img_bbox[1]) < 50) and \
                       ("Figure" in block_text or "Fig." in block_text or "Diagram" in block_text):
                        caption = block_text
                        break

                extracted_images.append({
                    "id": f"img_{image_count}",
                    "path": image_path,
                    "url": f"/api/images/{topic_id}/{image_filename}",
                    "page": i + 1,
                    "title": f"Textbook Figure - Page {i+1}",
                    "description": caption
                })
                image_count += 1

        return extracted_images

    def generate_image_embeddings(self, topic_id: str, extracted_images: List[Dict[str, Any]], embedding_service: EmbeddingService):
        """Generates embeddings for image descriptions for semantic search."""
        if not extracted_images: return

        topic_dir = os.path.join(self.base_image_dir, topic_id)
        descriptions = [img["description"] for img in extracted_images]
        
        embeddings = embedding_service.get_embeddings(descriptions)
        
        # Save embeddings and original metadata
        np.save(os.path.join(topic_dir, "embeddings.npy"), embeddings)
        with open(os.path.join(topic_dir, "metadata.json"), "w") as f:
            json.dump(extracted_images, f)
            
        print(f"Generated embeddings for {len(extracted_images)} images in topic {topic_id}")
