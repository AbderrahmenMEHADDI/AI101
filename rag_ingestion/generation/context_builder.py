"""
Context Builder
===============
Assembles the evidence block that will be passed to the LLM.

Responsibilities
----------------
- Accept a ranked list of retrieved chunks.
- Respect a strict MAX_CONTEXT_TOKENS budget.
- Deduplicate near-identical chunks.
- Preserve table and figure formatting.
- Clearly label each source so the LLM can cite it.

This class does NOT generate the prompt or call the LLM.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import tiktoken

from .. import config
from ..retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)

_tokenizer = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text, disallowed_special=()))


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContextSource:
    """One source entry included in the context."""
    rank: int
    chunk_id: str
    doc_id: str
    page: int
    chunk_type: str
    score: float
    content: str
    image_path: Optional[str] = None
    token_count: int = 0


@dataclass
class BuiltContext:
    """The assembled context ready for the prompt builder."""
    text: str                         # Full formatted context block
    sources: List[ContextSource] = field(default_factory=list)
    total_tokens: int = 0
    budget_tokens: int = 0
    dropped_count: int = 0            # Chunks dropped due to budget/dedup


# ─────────────────────────────────────────────────────────────────────────────
# Context Builder
# ─────────────────────────────────────────────────────────────────────────────

class ContextBuilder:
    """
    Builds a token-budgeted context block from retrieved chunks.

    Parameters
    ----------
    max_tokens : int
        Hard upper limit on context tokens.  Defaults to ``config.MAX_CONTEXT_TOKENS``.
    dedup_threshold : float
        Character-overlap ratio above which a chunk is considered a near-duplicate
        of one already included.  0.85 is a reasonable default.
    """

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        dedup_threshold: float = 0.85,
    ):
        self.max_tokens = max_tokens or config.MAX_CONTEXT_TOKENS
        self.dedup_threshold = dedup_threshold

    def build(self, results: List[RetrievalResult]) -> BuiltContext:
        """
        Assemble a context block from retrieved results.

        Chunks are taken in rank order (highest score first).
        A chunk is included only if it fits within the remaining token budget
        and is not a near-duplicate of an already-included chunk.

        Parameters
        ----------
        results : list[RetrievalResult]
            Retrieved and optionally reranked results (best first).

        Returns
        -------
        BuiltContext
            Formatted context string + source metadata.
        """
        # Sort by score descending (caller should already do this, but be safe)
        ordered = sorted(results, key=lambda r: r.score, reverse=True)

        included: List[ContextSource] = []
        blocks: List[str] = []
        used_tokens: int = 0
        dropped: int = 0
        seen_contents: List[str] = []

        for rank, result in enumerate(ordered, start=1):
            block = self._format_block(rank, result)
            block_tokens = _count_tokens(block)

            # Budget check
            if used_tokens + block_tokens > self.max_tokens:
                dropped += 1
                logger.debug(
                    "  Dropped chunk %s (tokens=%d, budget remaining=%d)",
                    result.chunk_id, block_tokens, self.max_tokens - used_tokens,
                )
                continue

            # Deduplication check
            if self._is_near_duplicate(result.content, seen_contents):
                dropped += 1
                logger.debug("  Dropped near-duplicate chunk %s", result.chunk_id)
                continue

            # Include
            blocks.append(block)
            used_tokens += block_tokens
            seen_contents.append(result.content)

            included.append(ContextSource(
                rank=rank,
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                page=result.page,
                chunk_type=result.chunk_type,
                score=round(result.score, 4),
                content=result.content,
                image_path=result.image_path,
                token_count=block_tokens,
            ))

        context_text = "\n\n".join(blocks)

        logger.debug(
            "Context built: %d sources, %d tokens / %d budget, %d dropped",
            len(included), used_tokens, self.max_tokens, dropped,
        )

        return BuiltContext(
            text=context_text,
            sources=included,
            total_tokens=used_tokens,
            budget_tokens=self.max_tokens,
            dropped_count=dropped,
        )

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_block(rank: int, result: RetrievalResult) -> str:
        """
        Format one chunk as a numbered source block.

        Example output
        --------------
        [Source 1]
        Document: paper.pdf | Page: 4 | Type: table | Score: 0.9123

        | Model | Accuracy |
        |-------|----------|
        | A     | 94.2%    |
        """
        header = (
            f"[Source {rank}]\n"
            f"Document: {result.doc_id} | "
            f"Page: {result.page + 1} | "
            f"Type: {result.chunk_type} | "
            f"Score: {result.score:.4f}"
        )
        return f"{header}\n\n{result.content}"

    @staticmethod
    def _is_near_duplicate(candidate: str, seen: List[str]) -> bool:
        """
        Simple character-overlap deduplication.

        Uses longest-common-subsequence approximation via set intersection
        on 4-grams — fast and good enough for RAG deduplication.
        """
        if not seen:
            return False

        def ngrams(text: str, n: int = 4) -> set:
            return {text[i : i + n] for i in range(len(text) - n + 1)}

        cand_grams = ngrams(candidate)
        if not cand_grams:
            return False

        for existing in seen:
            ex_grams = ngrams(existing)
            if not ex_grams:
                continue
            overlap = len(cand_grams & ex_grams) / max(len(cand_grams), len(ex_grams))
            if overlap >= 0.85:
                return True

        return False
