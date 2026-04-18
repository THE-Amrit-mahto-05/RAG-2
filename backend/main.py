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
from backend.services.semantic_cache import SemanticCache

import traceback # For detailed error logging

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

# Initialize directories (Moved outside project root to prevent StatReload loops)
BASE_DATA_DIR = os.path.expanduser("~/.edulevel/data")
UPLOAD_DIR = os.path.join(BASE_DATA_DIR, "uploads")
TOPICS_DIR = os.path.join(BASE_DATA_DIR, "topics")
IMAGE_DIR = os.path.join(BASE_DATA_DIR, "images")
CACHE_DIR = os.path.join(BASE_DATA_DIR, "cache")

for d in [UPLOAD_DIR, TOPICS_DIR, IMAGE_DIR, CACHE_DIR]:
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

@app.post("/api/upload", response_model=TopicMetadata)
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
    
    # --- STAGE 1: Decimal Pattern Discovery (e.g. 1.1, 11.2) ---
    for chunk in chunks:
        text = chunk["text"]
        for match in re.finditer(r'(?:^|\n)[ \t]*(?!Fig\.|Figure\s|Table\s)(\d+\.\d+(?:\.\d+)?)[ \t]+([^\n]{4,85})', text, re.IGNORECASE):
            section_num = match.group(1)
            raw_title = match.group(2).strip().rstrip('.')
            if section_num in seen or len(raw_title) < 4: continue
            toc.append({"section": section_num, "title": raw_title.title(), "page": chunk["page"]})
            seen.add(section_num)
    
    # Sort by section number correctly (handling arbitrary lengths)
    # --- STAGE 2: Formatting-Based Discovery (Headers with colons or ALL-CAPS) ---
    if len(toc) < 3:
        for chunk in chunks:
            text = chunk["text"]
            for match in re.finditer(r'(?:^|\n)\s*([A-Z][A-Za-z\s]{5,40}:|[A-Z]{5,40}(?:\s+[A-Z]{2,})*)\s*(?:\n|$)', text):
                raw_title = match.group(1).strip().rstrip(':')
                if raw_title.lower() in [t["title"].lower() for t in toc] or len(raw_title) < 5:
                    continue
                toc.append({
                    "section": str(len(toc) + 1),
                    "title": raw_title.title() if not raw_title.isupper() else raw_title,
                    "page": chunk["page"]
                })
                if len(toc) >= 10: break
            if len(toc) >= 10: break

    # --- STAGE 3: LLM Semantic Discovery (The "Smart" Fallback) ---
    if len(toc) < 3:
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

@app.post("/api/chat", response_model=ChatResponse)
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
