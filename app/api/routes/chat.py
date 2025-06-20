from fastapi import APIRouter, Depends, HTTPException
from langchain_core import messages
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
import logging

from app.models.database import get_db
from app.models.schemas import ChatRequest, ChatResponse, SearchResult, ChatManyRequest, ChatManyResponse
from app.services.vector_store import PgVectorStore
from app.services.RAGChatbot import RAGChatbot
from app.services.SimpleChatbot import SimpleChatbot

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

def get_simple_chatbot(db: Session = Depends(get_db)) -> SimpleChatbot:
    """Dependency để lấy simple chatbot instance"""
    session_id = id(db)  # Sử dụng id của db session làm key
    
    if session_id not in chatbot_instances:
        chatbot_instances[session_id] = SimpleChatbot()
    
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

@router.post("/simple-message", response_model=ChatResponse)
async def chat_message(
    request: ChatRequest,
    chatbot: SimpleChatbot = Depends(get_simple_chatbot)
):
    """
    Gửi tin nhắn đến chatbot
    
    Args:
        request: Yêu cầu chat
        chatbot: Instance của SimpleChatbot
        
    Returns:
        ChatResponse: Phản hồi từ SimpleChatbot
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Tin nhắn không được để trống")
        
        response, conversation_id = chatbot.chat(
            message=request.message,
            conversation_id=request.conversation_id
        )
        
        logger.info(f"Generated response for conversation {conversation_id}: {response}")
        return ChatResponse(
            response=response,
            sources=[],
            conversation_id=conversation_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi khi xử lý tin nhắn")

@router.post("/messages", response_model=ChatManyResponse)
async def chat_message(
    messages: ChatManyRequest,
    chatbot: RAGChatbot = Depends(get_chatbot)
):
    """
    Gửi tin nhắn đến chatbot
    
    Args:
        messages: Yêu cầu chat
        chatbot: Instance của RAGChatbot
        
    Returns:
        ChatManyResponse: Phản hồi từ chatbot
    """
    try:
        if not messages:
            raise HTTPException(status_code=400, detail="Tin nhắn không được để trống")
        
        responses = chatbot.chat_many(
            messages=messages.messages,
            conversation_id=messages.conversation_id
        )
        
        chat_responses = ChatManyResponse(
            responses=[],
            conversation_id=messages.conversation_id
        )
        
        for response, sources, conversation_id in responses:
            #logger.info(f"Generated response for conversation {conversation_id}: {response}")
            chat_responses.responses.append(ChatResponse(
                response=response,
                sources=sources,
                conversation_id=conversation_id
            ))

        return chat_responses
        
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