import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.services.embedding_service import EmbeddingService
from backend.services.image_matcher import ImageMatcher
from dotenv import load_dotenv

load_dotenv()
embedding_service = EmbeddingService(provider=os.getenv("EMBEDDING_PROVIDER", "local"))
matcher = ImageMatcher("backend/data/Sound")

response = "Longitudinal waves are a type of wave in which the particles of the medium move in a direction parallel to the direction of propagation of the disturbance [Source: Page 3]."

print("Testing exact match for 'longitudinal'...")
img = matcher.get_best_image("test_id", response, embedding_service, threshold=0.0)
if img: print("Matched:", img.title, img.url)
else: print("No match.")
