"""
Tests for the ContextBuilder.

Covers:
- Context never exceeds MAX_CONTEXT_TOKENS
- Sources ordered by score (descending)
- Near-duplicate chunks are dropped
- Dropped count reflects budget/dedup exclusions
- Table and figure content is preserved
"""

import pytest
from unittest.mock import MagicMock
from typing import List

from rag_ingestion.generation.context_builder import ContextBuilder, BuiltContext
from rag_ingestion.retrieval.retriever import RetrievalResult
from rag_ingestion import config


def _make_result(
    chunk_id: str,
    content: str,
    score: float = 0.9,
    page: int = 0,
    chunk_type: str = "text",
    doc_id: str = "doc.pdf",
) -> RetrievalResult:
    """Build a mock RetrievalResult."""
    result = MagicMock(spec=RetrievalResult)
    result.chunk_id = chunk_id
    result.doc_id = doc_id
    result.chunk_type = chunk_type
    result.content = content
    result.page = page
    result.score = score
    result.atomic = (chunk_type != "text")
    result.image_path = None
    return result


# ── Context never exceeds budget ──────────────────────────────────────────────

def test_context_never_exceeds_max_tokens():
    """Even with many chunks, the context must stay within budget."""
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Build chunks that together would exceed budget
    max_tok = config.MAX_CONTEXT_TOKENS
    # Each chunk ~100 tokens; far more than budget in total
    big_content = " ".join(["word"] * 80)
    results = [
        _make_result(f"c{i}", big_content, score=1.0 - i * 0.01)
        for i in range(100)
    ]

    builder = ContextBuilder(max_tokens=max_tok)
    ctx: BuiltContext = builder.build(results)

    actual_tokens = len(tokenizer.encode(ctx.text, disallowed_special=()))
    assert actual_tokens <= max_tok, (
        f"Context exceeded budget: {actual_tokens} > {max_tok}"
    )


def test_context_source_ordering():
    """Sources with higher scores should appear earlier in the context."""
    results = [
        _make_result("c1", "Low score content.", score=0.5),
        _make_result("c2", "High score content.", score=0.95),
        _make_result("c3", "Medium score content.", score=0.75),
    ]

    builder = ContextBuilder()
    ctx = builder.build(results)

    # Sources should be ordered highest score first
    scores = [s.score for s in ctx.sources]
    assert scores == sorted(scores, reverse=True)


def test_near_duplicate_dropped():
    """Near-identical chunks should be deduplicated."""
    content = "This is the exact same content repeated verbatim."
    results = [
        _make_result("c1", content, score=0.9),
        _make_result("c2", content, score=0.85),   # near-duplicate
        _make_result("c3", "Completely different content.", score=0.7),
    ]

    builder = ContextBuilder()
    ctx = builder.build(results)

    included_ids = [s.chunk_id for s in ctx.sources]
    # c1 should be included, c2 (duplicate) should be dropped, c3 should be included
    assert "c1" in included_ids
    assert "c2" not in included_ids
    assert "c3" in included_ids
    assert ctx.dropped_count >= 1


def test_empty_results():
    builder = ContextBuilder()
    ctx = builder.build([])
    assert ctx.text == ""
    assert ctx.sources == []
    assert ctx.total_tokens == 0


def test_table_content_preserved():
    """Table markdown must appear verbatim in the context."""
    table_md = "| Model | Accuracy |\n|-------|----------|\n| A     | 94.2%    |"
    results = [_make_result("t1", table_md, chunk_type="table", score=0.9)]

    builder = ContextBuilder()
    ctx = builder.build(results)

    assert table_md in ctx.text


def test_figure_content_preserved():
    """Figure descriptions must appear in context."""
    desc = "Figure Description:\nA scatter plot showing correlation."
    results = [_make_result("f1", desc, chunk_type="figure", score=0.85)]

    builder = ContextBuilder()
    ctx = builder.build(results)

    assert "scatter plot" in ctx.text


def test_budget_exact_fit():
    """Chunks that just fit should all be included."""
    # Use a very small budget
    builder = ContextBuilder(max_tokens=200)
    # Short content
    short = "Short."
    results = [_make_result(f"c{i}", short, score=1.0 - i * 0.1) for i in range(3)]
    ctx = builder.build(results)
    # All three should fit in 200 tokens
    assert len(ctx.sources) == 3


def test_source_metadata_in_context():
    """Source block must contain doc_id, page, and type."""
    result = _make_result("c1", "Some content.", page=4, chunk_type="text", doc_id="paper.pdf")
    builder = ContextBuilder()
    ctx = builder.build([result])

    assert "paper.pdf" in ctx.text
    assert "5" in ctx.text        # page + 1
    assert "text" in ctx.text
