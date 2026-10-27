"""
Pydantic models for database entities.
"""

from pydantic import BaseModel
from datetime import datetime


class Student(BaseModel):
    """Student profile."""
    id: str
    name: str
    email: str
    created_at: datetime


class QuizAttempt(BaseModel):
    """Record of a student's quiz attempt."""
    id: str
    student_id: str
    quiz_id: str
    score: float
    max_score: float
    submitted_at: datetime


class Conversation(BaseModel):
    """Chat conversation session."""
    id: str
    student_id: str
    messages: list[dict]
    created_at: datetime
    updated_at: datetime
