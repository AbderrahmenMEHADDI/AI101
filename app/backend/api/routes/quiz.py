"""
Quiz routes — Dynamic quiz and exam generation.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class QuizGenerateRequest(BaseModel):
    """Request to generate a quiz on a specific topic."""
    topic: str
    difficulty: str = "medium"  # easy, medium, hard
    num_questions: int = 5
    question_type: str = "mixed"  # mcq, short_answer, coding, mixed


class Question(BaseModel):
    """A single generated question."""
    id: int
    question: str
    question_type: str
    options: list[str] | None = None  # For MCQs
    points: int = 1


class QuizResponse(BaseModel):
    """Generated quiz."""
    quiz_id: str
    topic: str
    difficulty: str
    questions: list[Question]


@router.post("/generate", response_model=QuizResponse)
async def generate_quiz(request: QuizGenerateRequest):
    """Generate a quiz based on the specified topic and difficulty."""
    # TODO: Integrate LLM-based question generation
    return QuizResponse(
        quiz_id="placeholder",
        topic=request.topic,
        difficulty=request.difficulty,
        questions=[],
    )
