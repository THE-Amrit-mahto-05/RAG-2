"""
One-time script to regenerate image metadata for already-uploaded topics
so that better captions and embeddings are created from the improved image_processor.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.image_processor import ImageProcessor
from backend.services.embedding_service import EmbeddingService
from dotenv import load_dotenv

load_dotenv()

TOPICS_IMAGE_DIR = "backend/data/images"
UPLOAD_DIR = "backend/data/uploads"

embedding_service = EmbeddingService(provider=os.getenv("EMBEDDING_PROVIDER", "local"))
image_processor = ImageProcessor(TOPICS_IMAGE_DIR)

for topic_id in os.listdir(TOPICS_IMAGE_DIR):
    topic_path = os.path.join(TOPICS_IMAGE_DIR, topic_id)
    if not os.path.isdir(topic_path):
        continue
    
    # Find the corresponding uploaded PDF
    pdf_path = os.path.join(UPLOAD_DIR, f"{topic_id}.pdf")
    if not os.path.exists(pdf_path):
        print(f"  Skipping {topic_id}: no PDF found at {pdf_path}")
        continue
    
    print(f"\nReprocessing images for topic: {topic_id}")
    
    # Delete old image files but keep the directory structure
    for f in os.listdir(topic_path):
        file_path = os.path.join(topic_path, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    # Re-extract with improved captions
    images = image_processor.extract_images(pdf_path, topic_id)
    if images:
        image_processor.generate_image_embeddings(topic_id, images, embedding_service)
        print(f"  Done: {len(images)} images reprocessed with better captions.")
    else:
        print(f"  No images extracted.")

print("\nAll topics reprocessed.")
