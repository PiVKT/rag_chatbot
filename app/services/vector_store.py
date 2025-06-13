from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Tuple, Optional
import logging
from uuid import UUID

from app.models.database import Document, Chunk
from app.models.schemas import SearchResult
from app.services.embeddings import GeminiEmbeddings

logger = logging.getLogger(__name__)

class PgVectorStore:
    def __init__(self, db: Session):
        self.db = db
        self.embeddings = GeminiEmbeddings()
    
    def add_document(self, url: str, title: str, content: str, 
                    chunks: List[str], metadata: dict = None) -> UUID:
        """
        Thêm document và chunks vào vector store
        
        Args:
            url: URL của document
            title: Tiêu đề
            content: Nội dung gốc
            chunks: Danh sách chunks
            metadata: Metadata
            
        Returns:
            UUID: ID của document
        """
        try:
            # Tạo document
            doc = Document(
                url=url,
                title=title,
                content=content,
                meta_data=str(metadata or {})
            )
            self.db.add(doc)
            self.db.flush()  # Để lấy ID
            
            # Tạo embeddings cho chunks
            logger.info(f"Creating embeddings for {len(chunks)} chunks")
            embeddings = self.embeddings.embed_batch(chunks)
            
            # Lưu chunks
            for i, (chunk_content, embedding) in enumerate(zip(chunks, embeddings)):
                chunk = Chunk(
                    document_id=doc.id,
                    content=chunk_content,
                    embedding=embedding,
                    chunk_index=i,
                    meta_data=str(metadata or {})
                )
                self.db.add(chunk)
            
            self.db.commit()
            logger.info(f"Added document {doc.id} with {len(chunks)} chunks")
            
            return doc.id
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error adding document: {str(e)}")
            raise
    
    def semantic_search(self, query: str, max_results: int = 10, 
                       similarity_threshold: float = 0.7) -> List[SearchResult]:
        """
        Tìm kiếm semantic trong vector store
        
        Args:
            query: Câu truy vấn
            max_results: Số kết quả tối đa
            similarity_threshold: Ngưỡng similarity
            
        Returns:
            List[SearchResult]: Kết quả tìm kiếm
        """
        try:
            # Tạo query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Thực hiện vector search
            sql_query = text("""
                SELECT 
                    c.content,
                    c.embedding <=> :query_embedding as distance,
                    1 - (c.embedding <=> :query_embedding) as similarity,
                    d.url,
                    d.title
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE (1 - (c.embedding <=> :query_embedding)) >= :threshold
                ORDER BY c.embedding <=> :query_embedding
                LIMIT :max_results
            """)
            
            result = self.db.execute(sql_query, {
                'query_embedding': str(query_embedding),
                'threshold': similarity_threshold,
                'max_results': max_results
            })
            
            search_results = []
            for row in result:
                search_results.append(SearchResult(
                    content=row.content,
                    similarity=float(row.similarity),
                    document_url=row.url,
                    document_title=row.title
                ))
            
            logger.info(f"Found {len(search_results)} results for query: {query}")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            raise
    
    def get_document_by_url(self, url: str) -> Optional[Document]:
        """Lấy document theo URL"""
        return self.db.query(Document).filter(Document.url == url).first()
    
    def delete_document(self, document_id: UUID) -> bool:
        """Xóa document và chunks"""
        try:
            # Xóa chunks trước
            self.db.query(Chunk).filter(Chunk.document_id == document_id).delete()
            
            # Xóa document
            doc = self.db.query(Document).filter(Document.id == document_id).first()
            if doc:
                self.db.delete(doc)
                self.db.commit()
                return True
            
            return False
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting document: {str(e)}")
            raise
    
    def get_stats(self) -> dict:
        """Lấy thống kê về vector store"""
        try:
            doc_count = self.db.query(Document).count()
            chunk_count = self.db.query(Chunk).count()
            
            return {
                "total_documents": doc_count,
                "total_chunks": chunk_count,
                "avg_chunks_per_doc": round(chunk_count / doc_count, 2) if doc_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}