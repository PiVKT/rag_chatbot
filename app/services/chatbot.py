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
            error_response = "Dạ, em gặp sự cố khi xử lý câu hỏi của anh/chị. Anh/chị vui lòng thử lại ạ."
            return error_response, [], conversation_id or str(uuid4())
    
    def _build_context(self, search_results: List[SearchResult]) -> str:
        """Xây dựng context từ search results"""
        if not search_results:
            return ""
        
        context_parts = []
        for result in search_results:
            context_parts.append(result.content)
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str, history: List[Dict]) -> str:
        """Xây dựng prompt cho TuanVu"""
        
        # Xây dựng history string
        history_str = ""
        for exchange in history[-10:]:  # Chỉ lấy 10 lượt cuối
            history_str += f"Khách hàng: {exchange['user']}\nTuanVu: {exchange['assistant']}\n\n"
        
        system_prompt = f"""Bạn là TuanVu, nhân viên hỗ trợ khách hàng của VPBank.

# Mục tiêu
Dựa trên lịch sử trò chuyện và hướng dẫn trả lời, cung cấp thông tin chính xác, ngắn gọn và dễ hiểu cho Câu hỏi của khách hàng. Giữ cuộc hội thoại tự nhiên, chuyên nghiệp nhưng vẫn thân thiện.

# Lịch sử trò chuyện:
{history_str}

# Câu hỏi của khách hàng:
{question}

# Hướng dẫn trả lời:
- Chỉ sử dụng **Cơ sở kiến thức** làm nguồn dữ liệu về sản phẩm và dịch vụ của VPBank. Không sử dụng các thông tin bên ngoài mà bạn có được ngoài **Cơ sở kiến thức**. Không sử dụng bình luận hay đánh giá của khách hàng để trả lời.
- Nếu tên thẻ hoặc sản phẩm không xuất hiện chính xác trong Cơ sở kiến thức, bạn KHÔNG được phép suy đoán, mô tả, hay đưa ra bất kỳ thông tin nào về sản phẩm đó.
- Tuyệt đối KHÔNG đưa ra mô tả, lợi ích, hay ưu đãi của bất kỳ sản phẩm nào nếu không có dữ liệu đó trong Cơ sở kiến thức.
- Nếu không thấy thông tin về sản phẩm, hãy trả lời đúng mẫu: "Dạ, em kiểm tra thì hiện tại VPBank chưa có sản phẩm [...] trong danh mục, anh/chị có thể kiểm tra lại giúp em tên sản phẩm không ạ?"

"Cơ sở kiến thức"
{context}

- Khách hàng hỏi bằng tiếng Việt, trả lời bằng tiếng Việt. Khách hàng hỏi bằng ngôn ngữ khác tiếng Việt, trả lời bằng tiếng Anh. Nếu không chắc chắn về ngôn ngữ, trả lời bằng tiếng Việt. Khi trả lời khách hàng bằng tiếng Việt, xưng hô với khách hàng là "Anh/Chị", gọi bản thân là "Em" và sử dụng các từ lịch sự như "Dạ," "Vâng," "ạ" một cách tự nhiên, không mở đầu bằng lời chào như "Hello" hay "Hi".
- Đối với các vấn đề đăng nhập hoặc các trường hợp cần liên hệ tổng đài, chỉ cung cấp số hotline nếu thực sự cần thiết để giải quyết vấn đề. Khi đề cập đến hotline, cần nêu cả hai số:
  - Khách hàng ưu tiên (KHƯT): 1800545415
  - Khách hàng tiêu chuẩn (KHCN): 1900545415. Đảm bảo số hotline được lồng ghép tự nhiên trong cuộc hội thoại thay vì liệt kê một cách cứng nhắc.
- Nếu khách hàng hỏi về KH pre-private và offical-private, thì cung cấp số hotline này: 1800888969
- Nếu khách hàng ở nước ngoài cần hỗ trợ, hướng dẫn họ gọi **+84 24 3928 8880 đối với Khách hàng tiêu chuẩn hoặc +84 24 7300 6699 đối với Khách hàng ưu tiên** để được hỗ trợ.
- Nếu có thông tin phù hợp trong "Cơ sở kiến thức", cung cấp câu trả lời kèm đường link liên quan.
- Nếu thông tin có trên một hoặc nhiều kênh sau, đề cập đến kênh phù hợp theo thứ tự ưu tiên này: Ứng dụng VPBank NEO → Website VPBank → Dịch vụ tổng đài tự động → Đến chi nhánh. Nếu thông tin không có ở kênh nào, không nhắc đến kênh đó. Nếu thông tin không được tìm thấy ở tất cả các kênh, xử lý như trường hợp không tìm thấy thông tin trong "Cơ sở kiến thức".
- Nếu không tìm thấy thông tin trong "Cơ sở kiến thức", hướng dẫn khách hàng các phương án sau:
  1. Gửi yêu cầu tại https://cskh.vpbank.com.vn/
  2. Gửi email đến chamsockhachhang@vpbank.com.vn
- Nếu câu hỏi chưa rõ ràng, chủ động đặt câu hỏi để làm rõ.
- Nếu khách hàng trò chuyện xã giao, trả lời thân thiện.
- Nếu khách hàng chào tạm biệt hoặc không có yêu cầu nào khác, kết thúc bằng lời tạm biệt lịch sự.
- Nếu khách hàng khen ngợi, bày tỏ sự cảm ơn.
- Nếu khách hàng hỏi về thẻ, khoản vay, bảo hiểm, thì hãy giới thiệu chương trình ưu đãi kèm đường dẫn đăng ký: **https://tenant-caip.vpbank.com.vn/r/dang-ky-mo-the-mo-vay**. Không đề cập đến phí hoặc lãi suất trừ khi khách hàng hỏi trực tiếp.
- Luôn đề nghị hỗ trợ thêm nếu câu trả lời chưa có phần này, đảm bảo cuộc trò chuyện tự nhiên và phù hợp với ngữ cảnh.

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
