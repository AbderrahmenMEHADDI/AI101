"""
Application settings loaded from environment variables.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings."""

    APP_NAME: str = "AI101 - Adaptive Learning System"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS — Next.js dev server
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Database (if needed later)
    DATABASE_URL: str = "sqlite:///./ai101.db"

    # RAG
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_STORE_PATH: str = "rag_ingestion/storage"
    TOP_K_RESULTS: int = 5

    # LLM
    MODEL_PATH: str = "training/checkpoints/latest"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
