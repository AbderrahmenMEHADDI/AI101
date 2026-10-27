"""
Tests for QdrantStore.

Covers:
- Collection creation
- Upsert with pre-computed vectors
- Search / retrieval
- Dimension mismatch error
- Retrieval ordering (higher score first)

Uses a Qdrant in-memory client (:memory:) to avoid needing a running server.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from rag_ingestion.schema import Chunk
from rag_ingestion.storage.qdrant_store import QdrantStore

MOCK_DIM = 64


def _unit_vector(seed: int, dim: int = MOCK_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_chunk(n: int, chunk_type: str = "text") -> Chunk:
    return Chunk(
        chunk_id=f"chunk-{n:04d}",
        doc_id="test.pdf",
        chunk_type=chunk_type,
        content=f"Content of chunk {n}.",
        page=n,
        atomic=(chunk_type != "text"),
        token_count=10,
    )


@pytest.fixture
def store():
    """Create a QdrantStore backed by an in-memory Qdrant instance."""
    from qdrant_client import QdrantClient

    # Use in-memory Qdrant (no server required)
    client = QdrantClient(":memory:")

    # Patch the _get_client factory inside qdrant_store module
    with patch("rag_ingestion.storage.qdrant_store._get_client", return_value=client):
        # Patch embedder dimension
        with patch(
            "rag_ingestion.storage.qdrant_store.QdrantStore._get_vector_size",
            return_value=MOCK_DIM,
        ):
            s = QdrantStore("test_collection")
            yield s


def test_collection_creation(store):
    assert store.collection_exists()


def test_upsert_single_chunk(store):
    chunk = _make_chunk(0)
    vector = _unit_vector(0).reshape(1, -1)
    store.upsert_chunks([chunk], vector)

    results = store.search(query_vector=_unit_vector(0).tolist(), top_k=5)
    assert len(results) == 1
    assert results[0].payload["chunk_id"] == "chunk-0000"


def test_upsert_multiple_chunks(store):
    chunks = [_make_chunk(i) for i in range(5)]
    vectors = np.stack([_unit_vector(i) for i in range(5)])
    store.upsert_chunks(chunks, vectors)

    results = store.search(query_vector=_unit_vector(0).tolist(), top_k=10)
    assert len(results) == 5


def test_upsert_dimension_mismatch(store):
    chunk = _make_chunk(0)
    wrong_vector = np.random.rand(1, MOCK_DIM + 10).astype(np.float32)
    with pytest.raises(ValueError, match="dimension mismatch"):
        store.upsert_chunks([chunk], wrong_vector)


def test_upsert_length_mismatch(store):
    chunks = [_make_chunk(i) for i in range(3)]
    vectors = np.stack([_unit_vector(i) for i in range(2)])
    with pytest.raises(ValueError, match="Mismatch"):
        store.upsert_chunks(chunks, vectors)


def test_search_returns_scores(store):
    chunks = [_make_chunk(i) for i in range(3)]
    vectors = np.stack([_unit_vector(i) for i in range(3)])
    store.upsert_chunks(chunks, vectors)

    results = store.search(query_vector=_unit_vector(0).tolist(), top_k=3)
    for r in results:
        assert hasattr(r, "score")
        assert 0.0 <= r.score <= 1.01  # cosine can be slightly > 1 due to float precision


def test_search_ranking(store):
    """Best match should have highest score."""
    query_vec = _unit_vector(0)
    # chunk 0 will be most similar to query_vec (same seed)
    chunks = [_make_chunk(i) for i in range(4)]
    vectors = np.stack([_unit_vector(i) for i in range(4)])
    store.upsert_chunks(chunks, vectors)

    results = store.search(query_vector=query_vec.tolist(), top_k=4)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "Results not sorted by descending score"
    # chunk-0000 should be first
    assert results[0].payload["chunk_id"] == "chunk-0000"


def test_payload_stored_correctly(store):
    chunk = Chunk(
        chunk_id="payload-test",
        doc_id="doc.pdf",
        chunk_type="table",
        content="| A | B |",
        page=7,
        atomic=True,
        token_count=42,
        image_path=None,
        metadata={"quality": "good"},
    )
    vector = _unit_vector(99).reshape(1, -1)
    store.upsert_chunks([chunk], vector)

    results = store.search(query_vector=_unit_vector(99).tolist(), top_k=1)
    p = results[0].payload
    assert p["chunk_type"] == "table"
    assert p["page"] == 7
    assert p["token_count"] == 42
    assert p["metadata"]["quality"] == "good"


def test_chunk_type_filter(store):
    chunks = [_make_chunk(i, "text") for i in range(2)] + \
             [_make_chunk(i + 10, "table") for i in range(2)]
    vectors = np.stack([_unit_vector(i) for i in range(4)])
    store.upsert_chunks(chunks, vectors)

    text_results = store.search(
        query_vector=_unit_vector(0).tolist(),
        top_k=10,
        chunk_type_filter="text",
    )
    assert all(r.payload["chunk_type"] == "text" for r in text_results)
