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
                
                # Filter by resolution (Ignore tiny icons/bullets)
                if base_image["width"] < 150 or base_image["height"] < 150:
                    continue

                # Use Pixmap to enforce RGB color space (fixes blank CMYK images and Separation errors)
                pix = fitz.Pixmap(doc, xref)
                if pix.colorspace and pix.colorspace.name not in (fitz.csRGB.name, fitz.csGRAY.name):
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                
                image_ext = "png"
                image_filename = f"image_{image_count}.{image_ext}"
                image_path = os.path.join(topic_dir, image_filename)
                
                pix.save(image_path)
                pix = None # Free memory

                # HEURISTIC: Look for 'Figure X' or 'Diagram' near the image
                # In PyMuPDF, images have a 'rect' on the page. We check text blocks near that rect.
                caption = None
                img_bbox = page.get_image_bbox(img)  # Get image rect
                
                CAPTION_KEYWORDS = ("Figure", "Fig.", "Diagram", "Activity", "wave", "sound", "vibrat", "compress", "refract", "echo", "ultrasound")
                
                # Check blocks directly above or below the image rect (wider 100px)
                for b in blocks:
                    block_text = b[4].strip()
                    if (abs(b[1] - img_bbox[3]) < 100 or abs(b[3] - img_bbox[1]) < 100) and \
                       any(kw.lower() in block_text.lower() for kw in CAPTION_KEYWORDS):
                        caption = block_text[:300]  # limit to 300 chars
                        break
                
                # Fallback: collect nearby page text as description context
                if not caption:
                    nearby_texts = [b[4].strip() for b in blocks if abs(b[1] - img_bbox[3]) < 150 or abs(b[3] - img_bbox[1]) < 150]
                    caption = " ".join(nearby_texts[:3])[:300] if nearby_texts else f"Textbook diagram on page {i+1} related to sound waves, vibration, compression or propagation"

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
