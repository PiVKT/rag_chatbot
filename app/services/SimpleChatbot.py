import google.generativeai as genai
from typing import List, Dict, Optional
import logging
from uuid import uuid4

from app.config import settings

logger = logging.getLogger(__name__)

class SimpleChatbot:
    def __init__(self):
        genai.configure(api_key=settings.google_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.conversations: Dict[str, List[Dict]] = {}
    
    def chat(self, message: str, conversation_id: Optional[str] = None) -> tuple[str, str]:
        """
        Xử lý chat đơn giản không cần RAG
        
        Args:
            message: Tin nhắn từ user
            conversation_id: ID cuộc hội thoại
            
        Returns:
            tuple: (response, conversation_id)
        """
        try:
            # Tạo conversation_id mới nếu chưa có
            if not conversation_id:
                conversation_id = str(uuid4())
            
            # Lấy conversation history
            history = self.conversations.get(conversation_id, [])
            
            # Tạo prompt
            prompt = self._build_prompt(message, history)
            
            # Gọi Gemini
            response = self.model.generate_content(prompt)
            
            # Lưu vào conversation history
            self._update_conversation(conversation_id, message, response.text)
            
            logger.info(f"Generated response for conversation {conversation_id}")
            
            return response.text, conversation_id
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            error_response = "Xin lỗi, tôi gặp sự cố khi xử lý câu hỏi của bạn. Vui lòng thử lại."
            return error_response, conversation_id or str(uuid4())
    
    def _build_prompt(self, question: str, history: List[Dict]) -> str:
        """Xây dựng prompt cho chatbot đơn giản"""
        
        # Xây dựng history string
        history_str = ""
        for exchange in history[-10:]:  # Chỉ lấy 10 lượt cuối
            history_str += f"Người dùng: {exchange['user']}\nBot: {exchange['assistant']}\n\n"
        
        system_prompt = f"""Bạn là một trợ lý AI thông minh và hữu ích.

# Mục tiêu
Trả lời câu hỏi của người dùng một cách chính xác, hữu ích và thân thiện. Duy trì cuộc hội thoại tự nhiên và chuyên nghiệp.

# Lịch sử trò chuyện:
{history_str}

# Câu hỏi của người dùng:
{question}

# Hướng dẫn trả lời:
- Trả lời bằng tiếng Việt nếu người dùng hỏi bằng tiếng Việt, tiếng Anh nếu hỏi bằng tiếng Anh
- Cung cấp thông tin chính xác và hữu ích
- Giữ tone thân thiện và chuyên nghiệp
- Nếu không biết câu trả lời, hãy thành thật thừa nhận
- Nếu câu hỏi không rõ ràng, hãy yêu cầu làm rõ
- Trả lời ngắn gọn nhưng đầy đủ thông tin cần thiết
- Luôn sẵn sàng hỗ trợ thêm nếu cần

TRẢ LỜI:"""

        return system_prompt
    
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
    
    def get_all_conversations(self) -> Dict[str, List[Dict]]:
        """Lấy tất cả lịch sử hội thoại"""
        return self.conversations
