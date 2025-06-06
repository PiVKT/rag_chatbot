import google.generativeai as genai
from typing import List, Dict, Optional
import logging
from uuid import uuid4

from app.config import settings
from app.services.vector_store import PgVectorStore
from app.models.schemas import SearchResult

logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(self, vector_store: PgVectorStore):
        genai.configure(api_key=settings.google_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.vector_store = vector_store
        self.conversations: Dict[str, List[Dict]] = {}
    
    def chat(self, message: str, conversation_id: Optional[str] = None) -> tuple[str, List[SearchResult], str]:
        """
        Xử lý chat với RAG
        
        Args:
            message: Tin nhắn từ user
            conversation_id: ID cuộc hội thoại
            
        Returns:
            tuple: (response, sources, conversation_id)
        """
        try:
            # Tạo conversation_id mới nếu chưa có
            if not conversation_id:
                conversation_id = str(uuid4())
            
            # Tìm kiếm context từ vector store
            search_results = self.vector_store.semantic_search(
                query=message,
                max_results=settings.max_results,
                similarity_threshold=settings.similarity_threshold
            )
            
            # Tạo context từ search results
            context = self._build_context(search_results)
            
            # Lấy conversation history
            history = self.conversations.get(conversation_id, [])
            
            # Tạo prompt
            prompt = self._build_prompt(message, context, history)
            
            # Gọi Gemini
            response = self.model.generate_content(prompt)
            
            # Lưu vào conversation history
            self._update_conversation(conversation_id, message, response.text)
            
            logger.info(f"Generated response for conversation {conversation_id}")
            
            return response.text, search_results, conversation_id
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            error_response = "Xin lỗi, tôi gặp sự cố khi xử lý câu hỏi của bạn. Vui lòng thử lại."
            return error_response, [], conversation_id or str(uuid4())
    
    def _build_context(self, search_results: List[SearchResult]) -> str:
        """Xây dựng context từ search results"""
        if not search_results:
            return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"Nguồn {i} (từ {result.document_title}):\n{result.content}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str, history: List[Dict]) -> str:
        """Xây dựng prompt cho Gemini"""
        system_prompt = """
Bạn là một AI assistant thông minh, chuyên trả lời câu hỏi dựa trên thông tin được cung cấp.

HƯỚNG DẪN:
1. Trả lời dựa trên context được cung cấp
2. Nếu không có thông tin trong context, hãy nói rõ
3. Trả lời bằng tiếng Việt, chi tiết và dễ hiểu
4. Trích dẫn nguồn khi có thể
5. Nếu câu hỏi không liên quan đến context, hãy thông báo lịch sự

CONTEXT:
{context}

LỊCH SỬ HỘI THOẠI:
{history}

CÂU HỎI HIỆN TẠI: {question}

TRẢ LỜI:"""

        # Xây dựng history string
        history_str = ""
        for exchange in history[-3:]:  # Chỉ lấy 3 lượt cuối
            history_str += f"Người dùng: {exchange['user']}\nAI: {exchange['assistant']}\n\n"
        
        return system_prompt.format(
            context=context,
            history=history_str,
            question=question
        )
    
    def _update_conversation(self, conversation_id: str, user_message: str, assistant_response: str):
        """Cập nhật lịch sử hội thoại"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        self.conversations[conversation_id].append({
            "user": user_message,
            "assistant": assistant_response
        })
        
        # Giữ tối đa 10 lượt hội thoại
        if len(self.conversations[conversation_id]) > 10:
            self.conversations[conversation_id] = self.conversations[conversation_id][-10:]
    
    def clear_conversation(self, conversation_id: str):
        """Xóa lịch sử hội thoại"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Lấy lịch sử hội thoại"""
        return self.conversations.get(conversation_id, [])