"""
Tests for the Embedder.

Covers:
- Embedding dimension matches model
- Embeddings are normalized (unit vectors)
- Batch embedding shape
- Empty input handling
- Model name exposed
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# We mock SentenceTransformer to avoid downloading models during CI tests.
# The mock returns vectors of a known dimension.
# ─────────────────────────────────────────────────────────────────────────────

MOCK_DIM = 384
MOCK_MODEL_NAME = "mock-embedding-model"


def _make_unit_vectors(n: int, dim: int = MOCK_DIM) -> np.ndarray:
    """Return normalized float32 vectors."""
    rng = np.random.default_rng(0)
    vecs = rng.random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture(autouse=True)
def reset_embedder_singleton():
    """Reset the singleton before each test to allow fresh mocking."""
    import rag_ingestion.embeddings.embedder as mod
    original = mod.Embedder._instance
    mod.Embedder._instance = None
    yield
    mod.Embedder._instance = original


@pytest.fixture
def mock_st():
    """Patch SentenceTransformer globally before the Embedder is loaded."""
    with patch("rag_ingestion.embeddings.embedder.SentenceTransformer") as MockST:
        instance = MagicMock()
        instance.get_sentence_embedding_dimension.return_value = MOCK_DIM

        def _encode(texts, **kwargs):
            if isinstance(texts, str):
                return _make_unit_vectors(1)[0]
            return _make_unit_vectors(len(texts))

        instance.encode.side_effect = _encode
        MockST.return_value = instance
        MockST.return_value = instance
        yield MockST


def _get_fresh_embedder(mock_st_fixture):
    """Import and get a fresh Embedder instance with the mock active."""
    import importlib
    import rag_ingestion.embeddings.embedder as mod
    importlib.reload(mod)
    return mod.Embedder()


def test_embedding_dimension(mock_st):
    from rag_ingestion.embeddings.embedder import Embedder
    emb = Embedder()
    assert emb.get_dimension() == MOCK_DIM


def test_embeddings_are_normalized(mock_st):
    from rag_ingestion.embeddings.embedder import Embedder
    emb = Embedder()
    texts = ["The sky is blue.", "Transformers are powerful."]
    vectors = emb.embed_batch(texts)
    for v in vectors:
        norm = float(np.linalg.norm(v))
        assert abs(norm - 1.0) < 1e-5, f"Vector not normalized: norm={norm}"


def test_embed_batch_shape(mock_st):
    from rag_ingestion.embeddings.embedder import Embedder
    emb = Embedder()
    texts = ["a", "b", "c"]
    vecs = emb.embed_batch(texts)
    assert vecs.shape == (3, MOCK_DIM)


def test_embed_batch_empty(mock_st):
    from rag_ingestion.embeddings.embedder import Embedder
    emb = Embedder()
    vecs = emb.embed_batch([])
    assert vecs.shape == (0, MOCK_DIM)


def test_embed_single_empty_raises(mock_st):
    from rag_ingestion.embeddings.embedder import Embedder
    emb = Embedder()
    with pytest.raises(ValueError):
        emb.embed("")


def test_embed_batch_empty_string_raises(mock_st):
    from rag_ingestion.embeddings.embedder import Embedder
    emb = Embedder()
    with pytest.raises(ValueError):
        emb.embed_batch(["valid text", ""])


def test_embed_dtype_float32(mock_st):
    from rag_ingestion.embeddings.embedder import Embedder
    emb = Embedder()
    vecs = emb.embed_batch(["hello world"])
    assert vecs.dtype == np.float32


def test_model_name_exposed(mock_st):
    from rag_ingestion.embeddings.embedder import Embedder
    emb = Embedder()
    assert isinstance(emb.get_model_name(), str)
    assert len(emb.get_model_name()) > 0
