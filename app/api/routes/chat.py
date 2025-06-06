from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, List
import logging

from app.models.database import get_db
from app.models.schemas import ChatRequest, ChatResponse, SearchResult
from app.services.vector_store import PgVectorStore
from app.services.chatbot import RAGChatbot

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

# Cache chatbot instances
chatbot_instances: Dict[str, RAGChatbot] = {}

def get_chatbot(db: Session = Depends(get_db)) -> RAGChatbot:
    """Dependency để lấy chatbot instance"""
    session_id = id(db)  # Sử dụng id của db session làm key
    
    if session_id not in chatbot_instances:
        vector_store = PgVectorStore(db)
        chatbot_instances[session_id] = RAGChatbot(vector_store)
    
    return chatbot_instances[session_id]

@router.post("/message", response_model=ChatResponse)
async def chat_message(
    request: ChatRequest,
    chatbot: RAGChatbot = Depends(get_chatbot)
):
    """
    Gửi tin nhắn đến chatbot
    
    Args:
        request: Yêu cầu chat
        chatbot: Instance của RAGChatbot
        
    Returns:
        ChatResponse: Phản hồi từ chatbot
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Tin nhắn không được để trống")
        
        response, sources, conversation_id = chatbot.chat(
            message=request.message,
            conversation_id=request.conversation_id
        )
        
        return ChatResponse(
            response=response,
            sources=sources,
            conversation_id=conversation_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi khi xử lý tin nhắn")

@router.delete("/conversation/{conversation_id}")
async def clear_conversation(
    conversation_id: str,
    chatbot: RAGChatbot = Depends(get_chatbot)
):
    """Xóa lịch sử hội thoại"""
    try:
        chatbot.clear_conversation(conversation_id)
        return {"message": "Đã xóa lịch sử hội thoại"}
        
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi khi xóa lịch sử")

@router.get("/conversation/{conversation_id}")
async def get_conversation_history(
    conversation_id: str,
    chatbot: RAGChatbot = Depends(get_chatbot)
):
    """Lấy lịch sử hội thoại"""
    try:
        history = chatbot.get_conversation_history(conversation_id)
        return {"conversation_id": conversation_id, "history": history}
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi khi lấy lịch sử")