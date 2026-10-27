"""
RAG System Configuration
========================
All settings are read from environment variables.
Values in .env (project root) are loaded automatically.

Copy .env.example → .env and fill in your values.
Never hard-code secrets here.
"""

import os

# Load .env file if present (silently ignored if not found)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — use system env vars only


# ── OCR Settings ────────────────────────────────────────────────────────────
OCR_DPI: int = int(os.getenv("OCR_DPI", "300"))
OCR_LANG: str = os.getenv("OCR_LANG", "eng+fra+ara")
OCR_PSM: int = int(os.getenv("OCR_PSM", "3"))
MIN_OCR_CONFIDENCE: float = float(os.getenv("MIN_OCR_CONFIDENCE", "60"))

# ── Text Chunking Settings ──────────────────────────────────────────────────
MAX_CHUNK_TOKENS: int = int(os.getenv("MAX_CHUNK_TOKENS", "400"))
MIN_CHUNK_TOKENS: int = int(os.getenv("MIN_CHUNK_TOKENS", "80"))
OVERLAP_TOKENS: int = int(os.getenv("OVERLAP_TOKENS", "50"))
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))

# ── Embedding Model ─────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "") or None  # None = auto
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# ── Storage / Qdrant ────────────────────────────────────────────────────────
QDRANT_URL: str = os.getenv("QDRANT_URL", "https://f54e7fa7-f487-48ff-a914-2145d3407d42.sa-east-1-0.aws.cloud.qdrant.io")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6Y2I3MjhhYzgtZjg2Ny00ZDAzLThlODktMzVhYzE1ZmE1MjQ5In0.vnOvjmW0o2l0sORoFaNoMO--Ry4q2iLEqqEPCe6hLPg") or None  # None if empty
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_system_0")

# ── Figure Extraction ───────────────────────────────────────────────────────
FIGURES_DIR: str = os.getenv("FIGURES_DIR", "./data/figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

VISION_MODEL: str = os.getenv("VISION_MODEL", "qwen2.5vl:3b")
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")

# ── Generation / LLM ────────────────────────────────────────────────────────
GENERATOR_BACKEND: str = os.getenv("GENERATOR_BACKEND", "ollama")  # "ollama" or "gemini"
GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "qwen3:8b")
MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
GENERATION_TIMEOUT: int = int(os.getenv("GENERATION_TIMEOUT", "120"))

# Gemini (optional secondary backend)
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ── Retrieval Settings ──────────────────────────────────────────────────────
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "20"))
TOP_K_RERANK: int = int(os.getenv("TOP_K_RERANK", "5"))
FINAL_TOP_K: int = int(os.getenv("FINAL_TOP_K", "5"))

# ── Cross-Encoder Re-Ranker ──────────────────────────────────────────────────
CROSS_ENCODER_MODEL: str = os.getenv(
    "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ── Context Builder & Performance ──────────────────────────────────────────
MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "4000"))
ENABLE_THINKING: bool = os.getenv("ENABLE_THINKING", "false").lower() in ("true", "1", "yes")
ENABLE_WARMUP: bool = os.getenv("ENABLE_WARMUP", "true").lower() in ("true", "1", "yes")
