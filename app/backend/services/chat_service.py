"""
Chat service — Orchestrates RAG retrieval and LLM response generation.
"""


class ChatService:
    """Handles conversational tutoring logic."""

    def __init__(self):
        # TODO: Initialize RAG retriever and LLM
        pass

    async def generate_response(self, message: str, conversation_id: str | None = None) -> dict:
        """
        Generate a tutoring response grounded in course materials.

        Steps:
        1. Retrieve relevant document chunks via RAG
        2. Build prompt with retrieved context
        3. Generate LLM response
        4. Return response with source references
        """
        # TODO: Implement RAG + LLM pipeline
        return {
            "response": "Placeholder response",
            "sources": [],
            "conversation_id": conversation_id or "new",
        }
