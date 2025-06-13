from pydantic_settings import BaseSettings
from typing import Optional
import logging

class Settings(BaseSettings):
    # Database
    database_url: str
    
    # Gemini AI
    google_api_key: str
    
    # Application
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False
    
    # Embedding
    embedding_model: str = "models/embedding-001"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Vector search
    similarity_threshold: float = 0.7
    max_results: int = 10
    
    pgvector_extension: str = "vector"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

logging.getLogger('watchfiles.main').setLevel(logging.WARNING)
settings = Settings()