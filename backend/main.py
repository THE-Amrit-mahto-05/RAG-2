import os
import uuid
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.api.schema import ChatRequest, ChatResponse, TopicMetadata, Source, ImageInfo
from backend.core.processor import PDFProcessor
from backend.core.image_processor import ImageProcessor
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_store import VectorStore
from backend.services.retrieval_engine import RetrievalEngine
from backend.services.image_matcher import ImageMatcher

app = FastAPI(title="Edulevel RAG AI Tutor API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize directories
UPLOAD_DIR = "backend/data/uploads"
IMAGE_DIR = "backend/data/images"
PROCESSED_DIR = "backend/data/processed"
for d in [UPLOAD_DIR, IMAGE_DIR, PROCESSED_DIR]:
    os.makedirs(d, exist_ok=True)

# Serve images statically
app.mount("/api/images", StaticFiles(directory=IMAGE_DIR), name="images")

# Global instances (Loaded lazily or initialized here for simplicity)
embedding_service = EmbeddingService()
vector_store = VectorStore(dimension=embedding_service.dimension)
retrieval_engine = RetrievalEngine(embedding_service, vector_store)
image_matcher = ImageMatcher(IMAGE_DIR)
pdf_processor = PDFProcessor()
image_processor = ImageProcessor(IMAGE_DIR)

@app.get("/")
async def root():
    return {"message": "Edulevel AI Tutor API is running"}

@app.post("/api/upload", response_model=TopicMetadata)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    topic_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{topic_id}.pdf")
    
    # Save the PDF
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Process PDF: Extract text and chunk
    chunks = pdf_processor.process_pdf(file_path)
    
    # Generate embeddings and update vector store
    chunk_texts = [c.text for c in chunks]
    embeddings = embedding_service.get_embeddings(chunk_texts)
    vector_store.create_index(topic_id, embeddings, chunks)
    
    # Extract images and generate image embeddings
    images = image_processor.extract_images(file_path, topic_id)
    image_processor.generate_image_embeddings(topic_id, images, embedding_service)
    
    return TopicMetadata(
        id=topic_id,
        filename=file.filename,
        chunk_count=len(chunks),
        status="processed"
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    topic_id = request.topic_id
    question = request.question
    
    # 1. Retrieve Context
    retrieved_results = retrieval_engine.retrieve_context(topic_id, question)
    if not retrieved_results:
        return ChatResponse(
            answer="I couldn't find any relevant information in the document for that question.",
            sources=[],
            confidence=0.0
        )
    
    context = retrieval_engine.format_context_for_llm(retrieved_results)
    
    # 2. Generate LLM Answer (Mocking if no API key provided)
    # In a real scenario, we'd use Groq/OpenAI here.
    api_key = os.getenv("GROQ_API_KEY")
    
    if api_key:
        # Placeholder for real Groq call
        answer = f"[REAL LLM RESPONSE] Based on the context provided (Page {retrieved_results[0]['chunk'].page}), the document discussing {question} suggests..."
    else:
        # Mock Response for now
        main_chunk = retrieved_results[0]["chunk"]
        answer = f"According to the text on page {main_chunk.page}: '{main_chunk.text[:200]}...'"
    
    # 3. Match relevant image
    best_image = image_matcher.get_best_image(topic_id, answer, embedding_service)
    
    # 4. Format sources
    sources = [
        Source(chunk_id=res["chunk"].id, page=res["chunk"].page, similarity=res["similarity"])
        for res in retrieved_results
    ]
    
    return ChatResponse(
        answer=answer,
        image=best_image,
        sources=sources,
        confidence=sum(s.similarity for s in sources) / len(sources) if sources else 0.0
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
