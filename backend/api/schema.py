from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class TopicMetadata(BaseModel):
    id: str
    filename: str
    chunk_count: int
    status: str = "processed"

class Chunk(BaseModel):
    id: str
    text: str
    page: int
    metadata: Optional[Dict] = {}

class ChatRequest(BaseModel):
    topic_id: str
    question: str
    conversation_history: List[Dict] = []

class ImageInfo(BaseModel):
    url: str
    title: str
    description: str

class Source(BaseModel):
    chunk_id: str
    page: int
    similarity: float

class ChatResponse(BaseModel):
    answer: str
    image: Optional[ImageInfo] = None
    sources: List[Source]
    confidence: float
