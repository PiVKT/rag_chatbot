import google.generativeai as genai
from typing import List, Optional
import numpy as np
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)

class GeminiEmbeddings:
    def __init__(self):
        genai.configure(api_key=settings.google_api_key)
        self.model_name = settings.embedding_model
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def embed_text(self, text: str) -> List[float]:
        """
        Tạo embedding cho một text
        
        Args:
            text: Text cần embed
            
        Returns:
            List[float]: Vector embedding
        """
        try:
            if not text.strip():
                raise ValueError("Text is empty")
            
            # Truncate text if too long
            if len(text) > 10000:
                text = text[:10000]
            
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            
            return result['embedding']
            
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def embed_query(self, query: str) -> List[float]:
        """
        Tạo embedding cho query tìm kiếm
        
        Args:
            query: Query cần embed
            
        Returns:
            List[float]: Vector embedding
        """
        try:
            if not query.strip():
                raise ValueError("Query is empty")
            
            result = genai.embed_content(
                model=self.model_name,
                content=query,
                task_type="retrieval_query"
            )
            
            return result['embedding']
            
        except Exception as e:
            logger.error(f"Error creating query embedding: {str(e)}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        Tạo embedding cho nhiều texts
        
        Args:
            texts: Danh sách texts
            batch_size: Kích thước batch
            
        Returns:
            List[List[float]]: Danh sách embeddings
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch:
                try:
                    embedding = self.embed_text(text)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Failed to embed text: {str(e)}")
                    # Tạo zero vector như fallback
                    batch_embeddings.append([0.0] * 768)
            
            embeddings.extend(batch_embeddings)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        return embeddings