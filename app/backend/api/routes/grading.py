"""
Grading routes — Automated answer evaluation against marking rubrics.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class SubmissionRequest(BaseModel):
    """Student answer submission for grading."""
    quiz_id: str
    question_id: int
    answer: str
    answer_type: str = "text"  # text, code


class CriterionScore(BaseModel):
    """Score for a single rubric criterion."""
    criterion: str
    score: float
    max_score: float
    feedback: str


class GradingResponse(BaseModel):
    """Detailed grading result with rubric breakdown."""
    question_id: int
    total_score: float
    max_score: float
    breakdown: list[CriterionScore]
    overall_feedback: str


@router.post("/submit", response_model=GradingResponse)
async def submit_answer(request: SubmissionRequest):
    """
    Submit a student answer for automated grading.
    Returns a detailed score breakdown against the marking rubric.
    """
    # TODO: Integrate grading engine with rubric matching
    return GradingResponse(
        question_id=request.question_id,
        total_score=0.0,
        max_score=10.0,
        breakdown=[],
        overall_feedback="Grading engine not yet connected.",
    )
