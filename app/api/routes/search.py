from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import logging

from app.models.database import get_db
from app.models.schemas import SearchRequest, SearchResult
from app.services.vector_store import PgVectorStore

router = APIRouter(prefix="/search", tags=["search"])
logger = logging.getLogger(__name__)

@router.post("/semantic", response_model=List[SearchResult])
async def semantic_search(request: SearchRequest, db: Session = Depends(get_db)):
    """
    Tìm kiếm semantic trong vector store
    
    Args:
        request: Yêu cầu tìm kiếm
        db: Database session
        
    Returns:
        List[SearchResult]: Kết quả tìm kiếm
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query không được để trống")
        
        vector_store = PgVectorStore(db)
        results = vector_store.semantic_search(
            query=request.query,
            max_results=request.max_results,
            similarity_threshold=request.similarity_threshold
        )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi khi tìm kiếm")

@router.get("/stats")
async def get_search_stats(db: Session = Depends(get_db)):
    """Lấy thống kê về vector store"""
    try:
        vector_store = PgVectorStore(db)
        stats = vector_store.get_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi khi lấy thống kê")