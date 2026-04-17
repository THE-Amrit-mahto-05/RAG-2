import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.services.embedding_service import EmbeddingService
from backend.services.image_matcher import ImageMatcher
from dotenv import load_dotenv

load_dotenv()
embedding_service = EmbeddingService(provider=os.getenv("EMBEDDING_PROVIDER", "local"))
matcher = ImageMatcher("backend/data/Sound")

response = "When a bell is struck, it vibrates rapidly..."
keywords = ["bell", "vibration", "sound"]

print("\n--- Testing with Keywords ---")
img = matcher.get_best_image("test_id", f"{keywords[0]} {response[:100]}", embedding_service, threshold=0.0)
if img: print("Matched:", img.title)

print("\n--- Testing with Full Context ---")
img2 = matcher.get_best_image("test_id", response[:200], embedding_service, threshold=0.0)
if img2: print("Matched:", img2.title)
