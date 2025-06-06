from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class SemanticTextProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    def semantic_chunking(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Chia text thành chunks dựa trên ngữ nghĩa
        
        Args:
            text: Text cần chia
            metadata: Metadata đi kèm
            
        Returns:
            List[Document]: Danh sách chunks
        """
        if not text.strip():
            return []
        
        # Chia text cơ bản
        initial_chunks = self.text_splitter.split_text(text)
        
        if len(initial_chunks) <= 1:
            return [Document(page_content=text, meta_data=metadata or {})]
        
        # Tính toán semantic similarity
        semantic_chunks = self._merge_similar_chunks(initial_chunks)
        
        # Tạo Document objects
        documents = []
        for i, chunk in enumerate(semantic_chunks):
            chunk_metadata = (metadata or {}).copy()
            chunk_metadata.update({
                "chunk_index": i,
                "chunk_size": len(chunk),
                "total_chunks": len(semantic_chunks)
            })
            documents.append(Document(page_content=chunk, meta_data=chunk_metadata))
        
        return documents
    
    def _merge_similar_chunks(self, chunks: List[str], similarity_threshold: float = 0.6) -> List[str]:
        """
        Merge các chunks có similarity cao
        
        Args:
            chunks: Danh sách chunks
            similarity_threshold: Ngưỡng similarity
            
        Returns:
            List[str]: Chunks đã được merge
        """
        if len(chunks) <= 1:
            return chunks
        
        try:
            # Tính TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(chunks)
            
            # Tính cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            merged_chunks = []
            used_indices = set()
            
            for i in range(len(chunks)):
                if i in used_indices:
                    continue
                
                current_chunk = chunks[i]
                used_indices.add(i)
                
                # Tìm chunks tương tự để merge
                for j in range(i + 1, len(chunks)):
                    if j in used_indices:
                        continue
                    
                    if similarity_matrix[i][j] > similarity_threshold:
                        # Merge chunks nếu kích thước cho phép
                        if len(current_chunk) + len(chunks[j]) <= self.chunk_size * 1.5:
                            current_chunk += "\n\n" + chunks[j]
                            used_indices.add(j)
                
                merged_chunks.append(current_chunk)
            
            return merged_chunks
            
        except Exception as e:
            logger.warning(f"Semantic chunking failed, using basic chunking: {str(e)}")
            return chunks
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Trích xuất keywords từ text
        
        Args:
            text: Text cần trích xuất
            max_keywords: Số keywords tối đa
            
        Returns:
            List[str]: Danh sách keywords
        """
        try:
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=max_keywords * 2,
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Lấy top keywords
            top_indices = np.argsort(tfidf_scores)[::-1][:max_keywords]
            keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
            
            return keywords
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            return []