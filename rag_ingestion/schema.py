from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any


class Chunk(BaseModel):
    """
    Unified schema representing a chunk of data extracted from a document.
    Shared across text, tables, and figures.

    Fields
    ------
    chunk_id        : Unique identifier (UUID hex string).
    doc_id          : Source document identifier (usually the filename).
    chunk_type      : One of "text", "table", "figure".
    content         : The textual content that will be embedded and retrieved.
    page            : 0-based page index within the source document.
    atomic          : True for table/figure chunks that must not be split further.
    token_count     : Number of tokens in `content` (cl100k_base tokenizer).
    image_path      : Absolute path to the saved figure image (figure chunks only).
    parent_section  : Optional heading/section this chunk belongs to.
    metadata        : Arbitrary extra metadata (OCR stats, table quality, etc.).
    """

    chunk_id: str
    doc_id: str
    chunk_type: Literal["text", "table", "figure"]
    content: str                          # the text that will be embedded
    page: int
    atomic: bool                          # True for table/figure, False for text
    token_count: int = 0                  # populated by chunker/atomic_chunker
    image_path: Optional[str] = None
    parent_section: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
