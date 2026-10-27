"""
AI101 FastAPI Backend — Main Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.api.routes import chat, quiz, grading, analytics
from app.backend.core.config import settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        description="Adaptive Learning System API for the Introduction to AI Course",
        version=settings.APP_VERSION,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # CORS — allow the Next.js frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
    app.include_router(quiz.router, prefix="/api/quiz", tags=["Quiz"])
    app.include_router(grading.router, prefix="/api/grading", tags=["Grading"])
    app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])

    @app.get("/api/health", tags=["Health"])
    async def health_check():
        return {"status": "healthy", "version": settings.APP_VERSION}

    return app


app = create_app()
