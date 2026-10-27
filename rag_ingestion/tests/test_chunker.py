"""
Tests for the hybrid chunker.

Covers:
- Token counting
- Chunk never exceeds MAX_CHUNK_TOKENS
- Overlap is within budget (not additive)
- Single oversized sentence is handled
- Empty input
- token_count stored in each chunk
- Table/figure atomic chunks remain unsplit
"""

import pytest
from unittest.mock import patch
import numpy as np

from rag_ingestion.chunking.hybrid_chunker import (
    count_tokens,
    chunk_text,
    _split_oversized_sentence,
    _prepare_sentences,
)
from rag_ingestion.chunking.atomic_chunker import create_atomic_chunk
from rag_ingestion import config


# ── Token counting ────────────────────────────────────────────────────────────

def test_count_tokens_basic():
    assert count_tokens("Hello world") > 0


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_count_tokens_consistent():
    text = "The quick brown fox jumps over the lazy dog."
    assert count_tokens(text) == count_tokens(text)


# ── Oversized sentence splitting ──────────────────────────────────────────────

def test_oversized_sentence_splits():
    # Build a sentence that is definitely > 10 tokens
    long_sentence = " ".join(["word"] * 20)
    pieces = _split_oversized_sentence(long_sentence, max_tokens=5)
    assert len(pieces) > 1
    for piece in pieces:
        assert count_tokens(piece) <= 5


def test_short_sentence_not_split():
    sentence = "Short sentence."
    pieces = _split_oversized_sentence(sentence, max_tokens=50)
    assert pieces == [sentence]


# ── Chunk text ────────────────────────────────────────────────────────────────

def _make_mock_embeddings(n: int, dim: int = 384) -> np.ndarray:
    """Return unit vectors so cosine similarity is predictable."""
    rng = np.random.default_rng(42)
    vecs = rng.random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def test_chunk_text_empty_input():
    assert chunk_text("", "doc1", 0) == []


def test_chunk_text_blank_input():
    assert chunk_text("   \n  ", "doc1", 0) == []


def test_chunk_text_dict_input():
    """chunk_text should accept extractor dicts as well as plain strings."""
    sentences = " ".join(["This is a test sentence."] * 3)
    extractor_dict = {"text": sentences, "page_num": 0}

    with patch("rag_ingestion.chunking.hybrid_chunker.embed_batch",
               side_effect=lambda texts: _make_mock_embeddings(len(texts))):
        chunks = chunk_text(extractor_dict, "doc1", 0)

    assert isinstance(chunks, list)
    assert all(c.chunk_type == "text" for c in chunks)


def test_chunk_never_exceeds_max_tokens():
    """Hard constraint: no chunk may exceed MAX_CHUNK_TOKENS."""
    # Build text that is definitely bigger than MAX_CHUNK_TOKENS
    long_text = " ".join(["word"] * (config.MAX_CHUNK_TOKENS * 3))

    with patch("rag_ingestion.chunking.hybrid_chunker.embed_batch",
               side_effect=lambda texts: _make_mock_embeddings(len(texts))):
        chunks = chunk_text(long_text, "doc1", 0)

    assert chunks, "Expected at least one chunk"
    for chunk in chunks:
        assert chunk.token_count <= config.MAX_CHUNK_TOKENS, (
            f"Chunk exceeded MAX_CHUNK_TOKENS: {chunk.token_count} > {config.MAX_CHUNK_TOKENS}"
        )


def test_chunk_token_count_stored():
    """Every chunk must have a non-zero token_count."""
    text = "The quick brown fox. The lazy dog. Some more text here."

    with patch("rag_ingestion.chunking.hybrid_chunker.embed_batch",
               side_effect=lambda texts: _make_mock_embeddings(len(texts))):
        chunks = chunk_text(text, "doc1", 0)

    for chunk in chunks:
        assert chunk.token_count > 0


def test_chunk_overlap_within_budget():
    """
    Overlap must be part of (not additive to) MAX_CHUNK_TOKENS.

    We verify this by checking that no chunk has token_count >
    MAX_CHUNK_TOKENS even when overlap is active.
    """
    # Build enough text to force multiple chunks with overlap
    text = " ".join(["sentence number " + str(i) + "." for i in range(100)])

    with patch("rag_ingestion.chunking.hybrid_chunker.embed_batch",
               side_effect=lambda texts: _make_mock_embeddings(len(texts))):
        chunks = chunk_text(text, "doc1", 0)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.token_count <= config.MAX_CHUNK_TOKENS


def test_single_oversized_sentence_handled():
    """A sentence longer than MAX_CHUNK_TOKENS must not produce an unbounded chunk."""
    oversized = " ".join(["oversize"] * (config.MAX_CHUNK_TOKENS + 50))

    with patch("rag_ingestion.chunking.hybrid_chunker.embed_batch",
               side_effect=lambda texts: _make_mock_embeddings(len(texts))):
        chunks = chunk_text(oversized, "doc1", 0)

    assert chunks, "Expected at least one chunk from oversized input"
    for chunk in chunks:
        assert chunk.token_count <= config.MAX_CHUNK_TOKENS


# ── Atomic chunks ─────────────────────────────────────────────────────────────

def test_table_atomic_chunk():
    table_md = "| Col A | Col B |\n|-------|-------|\n| 1     | 2     |"
    chunk = create_atomic_chunk(table_md, "doc1", 3, "table")

    assert chunk.atomic is True
    assert chunk.chunk_type == "table"
    assert chunk.content == table_md
    assert chunk.token_count > 0


def test_figure_atomic_chunk():
    desc = "Figure Description:\nA bar chart showing model accuracy."
    chunk = create_atomic_chunk(desc, "doc1", 5, "figure", image_path="/tmp/fig.png")

    assert chunk.atomic is True
    assert chunk.chunk_type == "figure"
    assert chunk.image_path == "/tmp/fig.png"
    assert chunk.token_count > 0


def test_atomic_chunk_metadata():
    chunk = create_atomic_chunk(
        "content", "doc1", 0, "table",
        metadata={"rows": 5, "columns": 3, "quality": "good"}
    )
    assert chunk.metadata["rows"] == 5
    assert chunk.metadata["quality"] == "good"
