import os
import json
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.services.embedding_service import EmbeddingService
from dotenv import load_dotenv

load_dotenv()

SOUND_DIR = "backend/data/Sound"
JSON_PATH = os.path.join(SOUND_DIR, "images.json")
EMB_PATH = os.path.join(SOUND_DIR, "embeddings.npy")

embedding_service = EmbeddingService(provider=os.getenv("EMBEDDING_PROVIDER", "local"))

with open(JSON_PATH, "r") as f:
    images = json.load(f)

descriptions = [img["description"] for img in images]
embeddings = embedding_service.get_embeddings(descriptions)

np.save(EMB_PATH, embeddings)
print(f"Generated embeddings for {len(images)} static images.")
