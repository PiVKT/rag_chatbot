from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID

class WebsiteRequest(BaseModel):
    url: HttpUrl
    max_depth: int = 2
    max_pages: int = 10

class DocumentCreate(BaseModel):
    url: str
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = {}

class DocumentResponse(BaseModel):
    id: UUID
    url: str
    title: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class SearchRequest(BaseModel):
    query: str
    max_results: int = 5
    similarity_threshold: float = 0.7

class SearchResult(BaseModel):
    content: str
    similarity: float
    document_url: str
    document_title: str

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[SearchResult]
    conversation_id: str