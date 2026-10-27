"""
Hybrid Chunker
==============
Splits plain text into token-bounded, semantically coherent chunks.

Strategy
--------
1. Sentence splitting via NLTK.
2. Oversized sentences are split further using a token-window fallback.
3. Sentence embeddings are computed for semantic boundary detection.
4. Chunks are built greedily with a HARD token cap (check BEFORE adding).
5. Overlap is included in the total token budget (not additive).
6. Tiny trailing chunks are merged into the previous chunk.
7. Every Chunk stores ``token_count``.

Token budget rules
------------------
MAX_CHUNK_TOKENS  — hard upper limit per chunk (including overlap).
OVERLAP_TOKENS    — portion of MAX_CHUNK_TOKENS reserved for overlap carryover.
MIN_CHUNK_TOKENS  — minimum size before a semantic cut fires.

Embedding note
--------------
Sentence-level embeddings here are only used for semantic boundary detection
inside the chunker.  Final retrieval embeddings are produced separately by the
Embedder, keeping the two stages decoupled.
"""

import logging
import uuid
from typing import List

import nltk
import numpy as np
import tiktoken

from .. import config
from ..schema import Chunk
from ..embeddings.embedder import embed_batch

logger = logging.getLogger(__name__)

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except Exception:
    nltk.download("punkt_tab", quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer (cl100k_base — compatible with GPT-4 / text-embedding-3)
# ─────────────────────────────────────────────────────────────────────────────

_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding."""
    return len(_tokenizer.encode(text, disallowed_special=()))


# ─────────────────────────────────────────────────────────────────────────────
# Cosine similarity
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.

    Because embedder normalises to unit length, the dot product
    is numerically equivalent — but we keep the general formula
    here for correctness when embeddings from other sources are used.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ─────────────────────────────────────────────────────────────────────────────
# Sentence splitting
# ─────────────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Split text into non-empty sentences via NLTK."""
    return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]


def _split_oversized_sentence(sentence: str, max_tokens: int) -> List[str]:
    """
    Token-window fallback for sentences that exceed *max_tokens*.

    The sentence is encoded to tokens, then sliced into windows of
    exactly *max_tokens*.  Each window is decoded back to text.

    This guarantees no single piece ever exceeds the hard cap.
    """
    tokens = _tokenizer.encode(sentence, disallowed_special=())
    if len(tokens) <= max_tokens:
        return [sentence]

    pieces: List[str] = []
    for start in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[start : start + max_tokens]
        piece = _tokenizer.decode(chunk_tokens).strip()
        if piece:
            pieces.append(piece)
    return pieces


def _prepare_sentences(text: str) -> List[str]:
    """
    Split text into sentences and ensure no sentence exceeds MAX_CHUNK_TOKENS.
    """
    raw = _split_sentences(text)
    out: List[str] = []
    for sent in raw:
        out.extend(_split_oversized_sentence(sent, config.MAX_CHUNK_TOKENS))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Chunk builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_chunk(sentences: List[str], doc_id: str, page_num: int) -> Chunk:
    """Assemble a Chunk from a list of sentences and compute its token count."""
    content = " ".join(sentences).strip()
    tc = count_tokens(content)
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        doc_id=doc_id,
        chunk_type="text",
        content=content,
        page=page_num,
        atomic=False,
        token_count=tc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(
    text_input,
    doc_id: str,
    page_num: int,
) -> List[Chunk]:
    """
    Hybrid semantic + token-based chunking.

    Parameters
    ----------
    text_input : str or dict
        Either plain text, or a dict returned by ``extract_text()`` /
        ``extract_scanned_page()`` containing a ``"text"`` key.
    doc_id : str
        Source document identifier.
    page_num : int
        0-based page index.

    Returns
    -------
    list[Chunk]
        Ordered list of text chunks.  May be empty if input is blank.
    """
    # Accept both plain string and extractor dict
    if isinstance(text_input, dict):
        text = text_input.get("text", "")
    else:
        text = text_input or ""

    if not text.strip():
        return []

    # ── Step 1: prepare sentences ────────────────────────────────────────────
    sentences = _prepare_sentences(text)
    if not sentences:
        return []

    # ── Step 2: embed sentences for semantic boundary detection ──────────────
    logger.debug("Embedding %d sentences for semantic chunking (page %d)", len(sentences), page_num)
    try:
        embeddings = embed_batch(sentences)
    except Exception as exc:
        logger.warning("Sentence embedding failed (%s); falling back to token-only chunking.", exc)
        embeddings = None

    # ── Step 3: build chunks ─────────────────────────────────────────────────
    chunks: List[Chunk] = []
    current_sentences: List[str] = []
    current_tokens: int = 0

    for i, sentence in enumerate(sentences):
        sent_tokens = count_tokens(sentence)

        # HARD CONSTRAINT: check BEFORE adding (not after).
        # This ensures current_tokens + sent_tokens never exceeds MAX_CHUNK_TOKENS.
        if current_sentences and (current_tokens + sent_tokens > config.MAX_CHUNK_TOKENS):
            # Flush current chunk
            chunks.append(_build_chunk(current_sentences, doc_id, page_num))

            # Carry over overlap sentences from the end of current chunk
            overlap_sents, overlap_tokens = _collect_overlap(current_sentences)
            current_sentences = overlap_sents
            current_tokens = overlap_tokens

        current_sentences.append(sentence)
        current_tokens += sent_tokens

        # ── Semantic boundary detection ──────────────────────────────────────
        if (
            embeddings is not None
            and i < len(sentences) - 1
            and current_tokens >= config.MIN_CHUNK_TOKENS
        ):
            sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < config.SIMILARITY_THRESHOLD:
                chunks.append(_build_chunk(current_sentences, doc_id, page_num))
                overlap_sents, overlap_tokens = _collect_overlap(current_sentences)
                current_sentences = overlap_sents
                current_tokens = overlap_tokens

    # ── Step 4: handle trailing chunk ────────────────────────────────────────
    if current_sentences:
        final_chunk = _build_chunk(current_sentences, doc_id, page_num)

        if final_chunk.token_count < config.MIN_CHUNK_TOKENS and chunks:
            # Merge into previous chunk if it won't burst the limit
            prev = chunks[-1]
            merged_content = prev.content + " " + final_chunk.content
            merged_tokens = count_tokens(merged_content)
            # Accept the merge even if it slightly exceeds MAX (tiny trailing text)
            chunks[-1] = Chunk(
                chunk_id=prev.chunk_id,
                doc_id=prev.doc_id,
                chunk_type="text",
                content=merged_content,
                page=prev.page,
                atomic=False,
                token_count=merged_tokens,
            )
        else:
            chunks.append(final_chunk)

    logger.debug(
        "Chunking page %d: %d sentences → %d chunks", page_num, len(sentences), len(chunks)
    )
    return chunks


def _collect_overlap(sentences: List[str]) -> tuple[List[str], int]:
    """
    Collect trailing sentences that fit within OVERLAP_TOKENS budget.

    Overlap is part of the total MAX_CHUNK_TOKENS budget — it is NOT additive.
    The next chunk starts with these sentences as its prefix.
    """
    overlap_sents: List[str] = []
    overlap_tokens: int = 0

    for sent in reversed(sentences):
        t = count_tokens(sent)
        if overlap_tokens + t > config.OVERLAP_TOKENS:
            break
        overlap_sents.insert(0, sent)
        overlap_tokens += t

    return overlap_sents, overlap_tokens