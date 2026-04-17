import os
import uuid
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from backend.api.schema import ChatRequest, ChatResponse, TopicMetadata, Source
from backend.core.processor import PDFProcessor
from backend.core.image_processor import ImageProcessor
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_store import VectorStore
from backend.services.retrieval_engine import RetrievalEngine
from backend.services.image_matcher import ImageMatcher
from backend.services.llm_service import LLMService

# Load environment variables
load_dotenv()

app = FastAPI(title="Edulevel RAG AI Tutor API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize directories
UPLOAD_DIR = "backend/data/uploads"
TOPICS_DIR = "backend/data/topics"
IMAGE_DIR = "backend/data/images"
for d in [UPLOAD_DIR, TOPICS_DIR, IMAGE_DIR]:
    os.makedirs(d, exist_ok=True)

# Serve images statically
app.mount("/api/images", StaticFiles(directory=IMAGE_DIR), name="images")

# Shared services
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
embedding_service = EmbeddingService(provider=EMBEDDING_PROVIDER)

vector_store = VectorStore(dimension=embedding_service.dimension, base_path=TOPICS_DIR)
retrieval_engine = RetrievalEngine(embedding_service, vector_store)
image_matcher = ImageMatcher(IMAGE_DIR)
pdf_processor = PDFProcessor()
image_processor = ImageProcessor(IMAGE_DIR)

# PHASE 4 REFINEMENT: Initialize with dynamic provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
llm_service = LLMService(
    provider=LLM_PROVIDER,
    api_key=os.getenv(f"{LLM_PROVIDER.upper()}_API_KEY"),
    model=os.getenv("LLM_MODEL")
)

@app.get("/")
async def root():
    return {"message": f"Edulevel API: {EMBEDDING_PROVIDER} embeddings | {LLM_PROVIDER} LLM"}

@app.post("/api/upload", response_model=TopicMetadata)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    topic_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{topic_id}.pdf")
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # PDF Processing
    chunks = pdf_processor.process_pdf(file_path)
    
    # Store with Phase 2 hierarchical structure
    chunk_texts = [c.text for c in chunks]
    embeddings = embedding_service.get_embeddings(chunk_texts)
    vector_store.create_index(
        topic_id, 
        embeddings, 
        chunks, 
        metadata={"provider": EMBEDDING_PROVIDER}
    )
    
    # Image extraction
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
    
    # 1. Advanced Retrieval (Phase 3 improvements)
    retrieved_results = retrieval_engine.retrieve_context(topic_id, question)
    if not retrieved_results:
        return ChatResponse(
            answer="I couldn't find relevant textbook info. Try rephrasing!",
            sources=[],
            confidence=0.0
        )
    
    context = retrieval_engine.format_context_for_llm(retrieved_results)
    
    # 2. Refined Generation (Phase 4 improvements)
    generation = llm_service.generate_answer(
        question=question, 
        context=context, 
        history=request.conversation_history
    )
    
    answer = generation["answer"]
    keywords = generation["keywords"]
    
    # 3. Semantic Image Matching (using keywords from LLM)
    # We prioritize the first extracted keyword for matching
    best_image = None
    if keywords:
        # Hybrid match: use both LLM answer and primary keyword for context
        best_image = image_matcher.get_best_image(topic_id, f"{keywords[0]} {answer[:100]}", embedding_service)
    else:
        best_image = image_matcher.get_best_image(topic_id, answer, embedding_service)
    
    # 4. Sources with rich metadata
    sources = [
        Source(
            chunk_id=res["chunk"].id, 
            page=res["chunk"].page, 
            similarity=res["similarity"],
            match_metadata=res.get("meta")
        )
        for res in retrieved_results
    ]
    
    # Confidence calculation (vaguely groundedness weighted)
    confidence = sum(s.similarity for s in sources) / len(sources)
    if not generation.get("is_grounded", True):
        confidence *= 0.5 # De-prioritize ungrounded answers
    
    return ChatResponse(
        answer=answer,
        image=best_image,
        sources=sources,
        confidence=confidence
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
