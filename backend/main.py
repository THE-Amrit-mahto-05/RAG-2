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

# Add static mount for pre-provided assignment images
SOUND_DIR = "backend/data/Sound"
os.makedirs(SOUND_DIR, exist_ok=True)
app.mount("/api/static_images", StaticFiles(directory=SOUND_DIR), name="static_images")

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

@app.get("/api/toc/{topic_id}")
async def get_toc(topic_id: str):
    """Extracts real section headings from processed PDF chunks."""
    import json, re
    chunks_path = os.path.join(TOPICS_DIR, topic_id, "chunks.json")
    if not os.path.exists(chunks_path):
        raise HTTPException(status_code=404, detail="Topic not found")
    
    with open(chunks_path, "r") as f:
        chunks = json.load(f)
    
    # Combine all text to scan for headings
    toc = []
    seen = set()
    
    for chunk in chunks:
        text = chunk["text"]
        # Match patterns like "1.1 Introduction" or "11.2.1 Sound Waves Are Longitudinal Waves"
        # The title can be mixed-case words
        for match in re.finditer(r'\b(\d+\.\d+(?:\.\d+)?)\s+([A-Z][A-Za-z\s,\-]{3,60}?)(?=\s{1,3}[A-Z]|Activity|Question|Q\s|\n|\d{3,}|$)', text):
            section_num = match.group(1)
            raw_title = match.group(2).strip().rstrip('.')
            if section_num in seen or len(raw_title) < 4:
                continue
            # Skip if title is suspiciously all-caps noise  
            if raw_title.isupper() and len(raw_title) > 20:
                continue
            toc.append({
                "section": section_num,
                "title": raw_title.title(),
                "page": chunk["page"]
            })
            seen.add(section_num)
    
    # Sort by section number correctly (handling arbitrary lengths)
    try:
        toc.sort(key=lambda x: [int(n) for n in x["section"].split(".")])
    except Exception:
        pass
    
    # In case TOC extraction failed to find anything using strictly numbered patterns,
    # generate a generic page-based TOC framework
    if not toc:
        for chunk in chunks[:10]:
            if chunk["page"] not in [t.get("page") for t in toc]:
                toc.append({
                    "section": f"Page {chunk['page']}",
                    "title": f"Section on Page {chunk['page']}",
                    "page": chunk["page"]
                })
        # Keep it concise
        toc = toc[:5]
    

    
    return {"topic_id": topic_id, "toc": toc}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    topic_id = request.topic_id
    question = request.question
    
    # Validation: Ensure topic actually exists in memory/disk
    topic_path = os.path.join(TOPICS_DIR, topic_id)
    if not os.path.exists(topic_path):
        raise HTTPException(status_code=400, detail="Topic index not found. Please upload the file again.")
    
    # 1. Advanced Retrieval (Phase 3 improvements)
    retrieved_results = retrieval_engine.retrieve_context(topic_id, question)
    has_context = len(retrieved_results) > 0
    context = retrieval_engine.format_context_for_llm(retrieved_results) if has_context else ""
    
    # 2. Refined Generation (Phase 4 improvements)
    generation = llm_service.generate_answer(
        question=question, 
        context=context, 
        history=request.conversation_history,
        has_context=has_context
    )
    
    answer = generation["answer"]
    keywords = generation["keywords"]
    
    # 3. Dynamic Image Matching (No hardcoded subject links)
    best_image_info = None
    if keywords:
        best_image_info = image_matcher.get_best_image(topic_id, f"{' '.join(keywords)} {answer}", embedding_service, threshold=0.2)
    elif has_context:
        best_image_info = image_matcher.get_best_image(topic_id, answer, embedding_service, threshold=0.2)
        
    best_image = best_image_info.dict() if best_image_info else None
    
    # 4. Sources with rich metadata and raw text
    sources = retrieval_engine.get_sources_with_text(retrieved_results)
    
    # Confidence calculation
    confidence = 0.0
    if sources:
        confidence = sum(s.similarity for s in sources) / len(sources)
        if not generation.get("is_grounded", True):
            confidence *= 0.5
    
    return ChatResponse(
        answer=answer,
        image=best_image,
        sources=sources,
        confidence=confidence
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
