"""
Atomic Chunker
==============
Tables and figures are *atomic chunks* — they must never be split
arbitrarily because their meaning depends on their full structure.

Each table or figure becomes exactly one Chunk with ``atomic=True``.
"""

import uuid
from typing import Literal, Optional, Dict, Any

from ..schema import Chunk
from ..chunking.hybrid_chunker import count_tokens


def create_atomic_chunk(
    content: str,
    doc_id: str,
    page_num: int,
    chunk_type: Literal["table", "figure"],
    image_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Chunk:
    """
    Wrap a table or figure into a single, unsplit chunk.

    Parameters
    ----------
    content : str
        Full textual representation (markdown table or figure description).
    doc_id : str
        Source document identifier.
    page_num : int
        0-based page index.
    chunk_type : "table" or "figure"
        Content type for downstream filtering.
    image_path : str, optional
        Absolute path to the saved figure image (figure chunks only).
    metadata : dict, optional
        Extra metadata to carry in the chunk (e.g. table quality stats).

    Returns
    -------
    Chunk
        A single atomic Chunk ready for embedding and storage.
    """
    tc = count_tokens(content)

    return Chunk(
        chunk_id=str(uuid.uuid4()),
        doc_id=doc_id,
        chunk_type=chunk_type,
        content=content,
        page=page_num,
        atomic=True,
        token_count=tc,
        image_path=image_path,
        metadata=metadata or {},
    )
