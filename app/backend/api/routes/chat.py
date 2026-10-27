"""
Chat routes — Conversational tutoring chatbot powered by RAG.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ChatRequest(BaseModel):
    """Incoming chat message from the student."""
    message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    """Chatbot response with sources."""
    response: str
    sources: list[str] = []
    conversation_id: str


@router.post("/", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """
    Send a message to the AI tutoring chatbot.
    The response is grounded in course materials via RAG.
    """
    # TODO: Integrate RAG retrieval + LLM generation
    return ChatResponse(
        response="This endpoint will be connected to the RAG pipeline.",
        sources=[],
        conversation_id=request.conversation_id or "new-conversation",
    )
