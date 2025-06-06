from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.models.database import create_tables
from app.api.routes import scraping, search, chat
from app.utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events"""
    logger.info("Starting RAG Chatbot API")
    
    # Tạo tables
    try:
        create_tables()
        logger.info("Database tables created")
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
    
    yield
    
    logger.info("Shutting down RAG Chatbot API")

# Tạo FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="API cho hệ thống chatbot RAG với PgVector và Gemini AI",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production, nên giới hạn origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(scraping.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "RAG Chatbot API is running",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        from app.models.database import SessionLocal
        from app.services.vector_store import PgVectorStore
        
        # Kiểm tra database connection
        db = SessionLocal()
        try:
            vector_store = PgVectorStore(db)
            stats = vector_store.get_stats()
            db_status = "connected"
        except Exception as e:
            db_status = f"error: {str(e)}"
        finally:
            db.close()
        
        return {
            "status": "healthy",
            "database": db_status,
            "vector_store_stats": stats if db_status == "connected" else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug
    )