from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
import logging

from app.models.database import get_db
from app.models.schemas import WebsiteRequest, DocumentResponse
from app.services.web_scraper import WebScraper
from app.services.text_processor import SemanticTextProcessor
from app.services.vector_store import PgVectorStore

router = APIRouter(prefix="/scraping", tags=["scraping"])
logger = logging.getLogger(__name__)

@router.post("/scrape-website", response_model=dict)
async def scrape_website(
    request: WebsiteRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Scrape một website và lưu vào vector store
    
    Args:
        request: Thông tin website cần scrape
        background_tasks: Background tasks
        db: Database session
        
    Returns:
        dict: Thông tin về task
    """
    try:
        # Kiểm tra xem website đã được scrape chưa
        vector_store = PgVectorStore(db)
        existing_doc = vector_store.get_document_by_url(str(request.url))
        
        if existing_doc:
            raise HTTPException(
                status_code=400,
                detail=f"Website {request.url} đã được scrape trước đó"
            )
        
        # Thêm task vào background
        background_tasks.add_task(
            scrape_website_task,
            str(request.url),
            request.max_depth,
            request.max_pages,
            db
        )
        
        return {
            "message": f"Đã bắt đầu scrape website {request.url}",
            "status": "processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting scrape task: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi khi bắt đầu scrape website")

async def scrape_website_task(url: str, max_depth: int, max_pages: int, db: Session):
    """Background task để scrape website"""
    try:
        logger.info(f"Starting scrape task for {url}")
        
        # Khởi tạo services
        scraper = WebScraper()
        processor = SemanticTextProcessor()
        vector_store = PgVectorStore(db)
        
        # Scrape website
        scraped_content = scraper.scrape_website(url, max_depth, max_pages)
        
        # Xử lý từng trang
        for content in scraped_content:
            try:
                # Semantic chunking
                chunks = processor.semantic_chunking(content.content, content.metadata)
                chunk_texts = [chunk.page_content for chunk in chunks]
                
                # Lưu vào vector store
                vector_store.add_document(
                    url=content.url,
                    title=content.title,
                    content=content.content,
                    chunks=chunk_texts,
                    metadata=content.metadata
                )
                
                logger.info(f"Processed {content.url} with {len(chunk_texts)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {content.url}: {str(e)}")
                continue
        
        logger.info(f"Completed scrape task for {url}")
        
    except Exception as e:
        logger.error(f"Error in scrape task: {str(e)}")

@router.get("/documents", response_model=List[DocumentResponse])
async def get_documents(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    """Lấy danh sách documents đã scrape"""
    try:
        from app.models.database import Document
        documents = db.query(Document).offset(skip).limit(limit).all()
        return documents
        
    except Exception as e:
        logger.error(f"Error getting documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi khi lấy danh sách documents")

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Xóa một document"""
    try:
        from uuid import UUID
        vector_store = PgVectorStore(db)
        success = vector_store.delete_document(UUID(document_id))
        
        if success:
            return {"message": "Document đã được xóa"}
        else:
            raise HTTPException(status_code=404, detail="Không tìm thấy document")
            
    except ValueError:
        raise HTTPException(status_code=400, detail="ID không hợp lệ")
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi khi xóa document")