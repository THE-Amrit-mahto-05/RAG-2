import os
import json
import fitz # PyMuPDF
import numpy as np
from PIL import Image, ImageStat
from typing import List, Dict, Any
from backend.services.embedding_service import EmbeddingService

class ImageProcessor:
    def __init__(self, base_image_dir: str = "backend/data/images"):
        self.base_image_dir = base_image_dir
        os.makedirs(base_image_dir, exist_ok=True)

    def extract_images(self, pdf_path: str, topic_id: str) -> List[Dict[str, Any]]:
        """Extracts significant images (raster and vector) from PDF and attempts to find matching captions."""
        doc = fitz.open(pdf_path)
        topic_dir = os.path.join(self.base_image_dir, topic_id)
        os.makedirs(topic_dir, exist_ok=True)
        
        extracted_images = []
        image_count = 0

        CAPTION_KEYWORDS = ("Figure", "Fig.", "Diagram", "Activity", "wave", "sound", "vibrat", "compress", "refract", "echo", "ultrasound")

        for i in range(len(doc)):
            page = doc[i]
            blocks = page.get_text("blocks")
            
            # --- STAGE 1: Standard Raster Extraction ---
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                
                if base_image["width"] < 150 or base_image["height"] < 150:
                    continue

                pix = fitz.Pixmap(doc, xref)
                if pix.colorspace and pix.colorspace.name not in (fitz.csRGB.name, fitz.csGRAY.name):
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                
                # Filter solid colors/blank images
                pil_mode = "RGB" if pix.n == 3 else ("RGBA" if pix.n == 4 else "L")
                try:
                    pil_img = Image.frombytes(pil_mode, [pix.width, pix.height], pix.samples)
                    stat = ImageStat.Stat(pil_img.convert("L"))
                    if stat.stddev[0] < 10.0:
                        continue
                except Exception:
                    pass
                    
                image_filename = f"image_{image_count}.png"
                image_path = os.path.join(topic_dir, image_filename)
                pix.save(image_path)
                img_bbox = page.get_image_bbox(img)
                
                caption = self._find_caption(blocks, img_bbox, CAPTION_KEYWORDS, i)
                extracted_images.append({
                    "id": f"img_{image_count}",
                    "path": image_path,
                    "url": f"/api/images/{topic_id}/{image_filename}",
                    "page": i + 1,
                    "title": f"Textbook Figure - Page {i+1}",
                    "description": caption
                })
                image_count += 1
                pix = None

            # --- STAGE 2: Vector Graphics (Drawings) Clustering ---
            drawings = page.get_drawings()
            if drawings:
                page_area = page.rect.width * page.rect.height
                # Group nearby drawings into clusters
                clusters = [] # list of {"rect": Rect, "count": int}
                
                for d in drawings:
                    rect = fitz.Rect(d["rect"])
                    # Ignore tiny lines or giant background rectangles (>20% page area)
                    if rect.width < 10 and rect.height < 10: continue
                    if (rect.width * rect.height) > (page_area * 0.20): continue
                    
                    added = False
                    for cluster_info in clusters:
                        cluster = cluster_info["rect"]
                        c1 = ((cluster.x0 + cluster.x1) / 2, (cluster.y0 + cluster.y1) / 2)
                        c2 = ((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
                        dist = ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5
                        
                        # Use a tighter distance for clustering (40px)
                        if dist < 40 or cluster.intersects(rect): 
                            cluster.include_rect(rect)
                            cluster_info["count"] += 1
                            added = True
                            break
                    if not added:
                        clusters.append({"rect": rect, "count": 1})

                for cluster_info in clusters:
                    rect = cluster_info["rect"]
                    # Must contain multiple components to be a "diagram" 
                    # OR be a reasonably sized individual item
                    if cluster_info["count"] < 3 and (rect.width < 100 or rect.height < 60):
                        continue
                        
                    if rect.width / rect.height > 10 or rect.height / rect.width > 10: 
                        continue # Skip long separator lines
                    
                    # Ensure we don't grab something massive
                    if rect.width > page.rect.width * 0.9 or rect.height > page.rect.height * 0.8:
                        continue
                    
                    # Render the cluster
                    pix = page.get_pixmap(clip=rect, dpi=150)
                    
                    # Variance check for vector renders too
                    try:
                        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        stat = ImageStat.Stat(pil_img.convert("L"))
                        if stat.stddev[0] < 12.0: # Vectors need slightly higher variance to be useful
                            continue
                    except: pass

                    image_filename = f"image_{image_count}.png"
                    image_path = os.path.join(topic_dir, image_filename)
                    pix.save(image_path)
                    
                    caption = self._find_caption(blocks, rect, CAPTION_KEYWORDS, i)
                    extracted_images.append({
                        "id": f"img_{image_count}",
                        "path": image_path,
                        "url": f"/api/images/{topic_id}/{image_filename}",
                        "page": i + 1,
                        "title": f"Scientific Diagram - Page {i+1}",
                        "description": caption
                    })
                    image_count += 1
                    pix = None

        return extracted_images

    def _find_caption(self, blocks, bbox, keywords, page_num):
        """Helper to find captions or nearby text for an image/diagram."""
        caption = None
        for b in blocks:
            block_text = b[4].strip()
            # Check proximity (within 100px vertically)
            if (abs(b[1] - bbox.y1) < 100 or abs(b[3] - bbox.y0) < 100) and \
               any(kw.lower() in block_text.lower() for kw in keywords):
                caption = block_text[:300]
                break
        
        if not caption:
            # Fallback to general nearby text (wider range)
            nearby_texts = [b[4].strip() for b in blocks if abs(b[1] - bbox.y1) < 150 or abs(b[3] - bbox.y0) < 150]
            if nearby_texts:
                caption = " ".join(nearby_texts[:3])[:300]
            else:
                caption = f"Educational diagram on page {page_num + 1} illustrating core scientific concepts"
        
        return caption

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
