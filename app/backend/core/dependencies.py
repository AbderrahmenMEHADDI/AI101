"""
Dependency injection utilities for FastAPI.
"""

from functools import lru_cache

from app.backend.core.config import Settings


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
