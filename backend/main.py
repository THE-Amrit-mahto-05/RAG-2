import os
import uuid
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from api.schema import ChatRequest, ChatResponse, TopicMetadata, Source
from core.processor import PDFProcessor
from core.image_processor import ImageProcessor
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore
from services.retrieval_engine import RetrievalEngine
from services.image_matcher import ImageMatcher
from services.llm_service import LLMService
from services.semantic_cache import SemanticCache

import traceback # For detailed error logging

# Load environment variables (Forced override to pick up settings from .env)
load_dotenv(override=True)

app = FastAPI(title="Edulevel RAG AI Tutor API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize directories (Moved outside project root to prevent StatReload loops)
BASE_DATA_DIR = os.path.expanduser("~/.edulevel/data")
UPLOAD_DIR = os.path.join(BASE_DATA_DIR, "uploads")
TOPICS_DIR = os.path.join(BASE_DATA_DIR, "topics")
IMAGE_DIR = os.path.join(BASE_DATA_DIR, "images")
CACHE_DIR = os.path.join(BASE_DATA_DIR, "cache")

for d in [UPLOAD_DIR, TOPICS_DIR, IMAGE_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# Serve images statically
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# Add static mount for pre-provided assignment images
SOUND_DIR = "backend/data/Sound"
os.makedirs(SOUND_DIR, exist_ok=True)
app.mount("/static_images", StaticFiles(directory=SOUND_DIR), name="static_images")

# Shared services
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
embedding_service = EmbeddingService(provider=EMBEDDING_PROVIDER)

vector_store = VectorStore(dimension=embedding_service.dimension, base_path=TOPICS_DIR)
retrieval_engine = RetrievalEngine(embedding_service, vector_store)
image_matcher = ImageMatcher(IMAGE_DIR)
pdf_processor = PDFProcessor()
image_processor = ImageProcessor(IMAGE_DIR)
semantic_cache = SemanticCache(data_dir=CACHE_DIR, max_size=60)

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

@app.post("/upload", response_model=TopicMetadata)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    topic_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{topic_id}.pdf")
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # PDF Processing
        chunks = pdf_processor.process_pdf(file_path)
        
        # Store embeddings
        chunk_texts = [c.text for c in chunks]
        embeddings = embedding_service.get_embeddings(chunk_texts)
        vector_store.create_index(
            topic_id, 
            embeddings, 
            chunks, 
            metadata={"provider": EMBEDDING_PROVIDER}
        )
        
        # Image extraction (can take time)
        images = image_processor.extract_images(file_path, topic_id)
        image_processor.generate_image_embeddings(topic_id, images, embedding_service)
        
        return TopicMetadata(
            id=topic_id,
            filename=file.filename,
            chunk_count=len(chunks),
            status="processed"
        )
    except Exception as e:
        print(f"CRITICAL ERROR DURING UPLOAD: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/toc/{topic_id}")
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
    
    # --- STAGE 1: Numbered Pattern Discovery (e.g. 1., 1.1, 11.2) ---
    for chunk in chunks:
        text = chunk["text"]
        # Pattern: Start of line, Optional whitespace, Number (e.g. 1 or 1.1), Period (mandatory for single digits, optional for decimals), Space, Title
        for match in re.finditer(r'(?:^|\n)[ \t]*(?!Fig\.|Figure\s|Table\s)(\d+(?:\.\d+)*)\.?[ \t]+([^\n]{4,85})', text, re.IGNORECASE):
            section_num = match.group(1)
            raw_title = match.group(2).strip().rstrip('.')
            # Avoid picking up lowercase sentence starts that look like numbers
            if section_num in seen or len(raw_title) < 4 or raw_title[0].islower(): continue
            toc.append({"section": section_num, "title": raw_title.title(), "page": chunk["page"]})
            seen.add(section_num)
    
    # --- STAGE 2: Formatting-Based Discovery (Headers with colons or ALL-CAPS) ---
    # We always run this to find major chapter divisions, even if Stage 1 found some
    for chunk in chunks:
        text = chunk["text"]
        # Pattern: ALL-CAPS headers or Titles ending with colons
        for match in re.finditer(r'(?:^|\n)\s*([A-Z][A-Za-z\s]{5,40}:|[A-Z]{5,40}(?:\s+[A-Z]{2,})*)\s*(?:\n|$)', text):
            raw_title = match.group(1).strip().rstrip(':')
            if raw_title.lower() in [t["title"].lower() for t in toc] or len(raw_title) < 5:
                continue
            toc.append({
                "section": str(len(toc) + 1),
                "title": raw_title.title() if not raw_title.isupper() else raw_title,
                "page": chunk["page"]
            })
            if len(toc) >= 50: break
        if len(toc) >= 50: break

    # --- STAGE 3: LLM Semantic Discovery (The "Smart" Fallback) ---
    if len(toc) < 5:
        print("DEBUG: TOC sparse. Triggering LLM-assisted Discovery...")
        try:
            combined_text = "\n".join([c["text"] for c in chunks[:5]])
            llm_toc = llm_service.extract_toc(combined_text)
            if llm_toc:
                toc = llm_toc
        except Exception as e:
            print(f"WARNING: LLM-assisted TOC discovery failed (possible rate limit): {e}")
            # Continue with existing 'toc' (even if sparse or empty) instead of crashing with 500 error

    # Sort and return
    try:
        toc.sort(key=lambda x: [int(n) if str(n).isdigit() else 0 for n in str(x.get("section", "0")).split(".")])
    except Exception:
        pass
    

    
    return {"topic_id": topic_id, "toc": toc}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    topic_id = request.topic_id
    question = request.question
    
    # Validation: Ensure topic actually exists in memory/disk
    topic_path = os.path.join(TOPICS_DIR, topic_id)
    if not os.path.exists(topic_path):
        raise HTTPException(status_code=400, detail="Topic index not found. Please upload the file again.")
        
    # 0. Check Semantic Cache
    cached_data = semantic_cache.get(topic_id, question, embedding_service)
    if cached_data:
        return ChatResponse(**cached_data)
    
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
            
    # Cache the generated response
    response_data = {
        "answer": answer,
        "image": best_image,
        "sources": [s.dict() for s in sources],
        "confidence": confidence
    }
    semantic_cache.put(topic_id, question, response_data, embedding_service)
    
    return ChatResponse(
        answer=answer,
        image=best_image,
        sources=sources,
        confidence=confidence
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
