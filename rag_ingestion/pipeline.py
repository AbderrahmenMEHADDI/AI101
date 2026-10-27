"""
Ingestion Pipeline
==================
Orchestrates the full document ingestion flow:

    PDF → load → classify regions → extract content → chunk → embed → store

Public entry points
-------------------
ingest_document(filepath, collection_name, doc_id)
    Full pipeline including embedding and Qdrant upsert.

process_document(filepath, doc_id)
    Extraction + chunking only (returns chunks, no storage).
    Kept for backwards compatibility and testing.
"""

import logging
import os
from typing import List, Optional, Tuple

from .schema import Chunk
from .loaders.router import get_page_count
from .classifiers.content_classifier import classify_page_content
from .extractors.text_extractor import extract_text
from .extractors.table_extractor import extract_table
from .extractors.figure_extractor import extract_figure
from .extractors.ocr_extractor import extract_scanned_page
from .chunking.hybrid_chunker import chunk_text
from .chunking.atomic_chunker import create_atomic_chunk

logger = logging.getLogger(__name__)

BBox = Tuple[float, float, float, float]


def process_document(filepath: str, doc_id: str) -> List[Chunk]:
    """
    Run the full extraction + chunking pipeline for a single PDF.

    This function does NOT embed or store — it just produces Chunks.
    Use ``ingest_document()`` for the complete pipeline.

    Parameters
    ----------
    filepath : str
        Path to the PDF file.
    doc_id : str
        Unique document identifier (usually the filename).

    Returns
    -------
    list[Chunk]
        All chunks extracted from the document.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Document not found: {filepath}")

    num_pages = get_page_count(filepath)
    logger.info("Processing '%s' (%d pages)", os.path.basename(filepath), num_pages)

    all_chunks: List[Chunk] = []

    for page_num in range(num_pages):
        logger.debug("  Page %d/%d …", page_num + 1, num_pages)
        try:
            page_chunks = _process_page(filepath, doc_id, page_num)
            all_chunks.extend(page_chunks)
        except Exception as exc:
            # Page-level failure must not destroy the entire ingestion
            logger.error(
                "  [ERROR] Page %d failed: %s — skipping.", page_num, exc, exc_info=True
            )

    logger.info(
        "Finished '%s': %d chunks produced.", os.path.basename(filepath), len(all_chunks)
    )
    return all_chunks


def _process_page(
    filepath: str,
    doc_id: str,
    page_num: int,
) -> List[Chunk]:
    """
    Extract and chunk all content regions on a single page.

    Returns a list of Chunk objects (may be empty if the page has no usable content).
    """
    # 1. Classify regions
    regions = classify_page_content(filepath, page_num)

    chunks: List[Chunk] = []
    table_bboxes: List[BBox] = []
    figure_bboxes: List[BBox] = []

    # ── First pass: extract tables and figures ────────────────────────────────
    # We do these first so we can pass their bboxes as exclusions to the text
    # extractor, preventing content duplication.

    for region in regions:
        rtype = region["type"]
        bbox: BBox = tuple(region["bbox"])

        if rtype == "table":
            table_result = None
            try:
                table_result = extract_table(filepath, page_num, bbox)
            except Exception as exc:
                logger.warning("Table extraction failed page %d: %s", page_num, exc)

            if table_result:
                table_bboxes.append(bbox)
                chunk = create_atomic_chunk(
                    content=table_result["text"],
                    doc_id=doc_id,
                    page_num=page_num,
                    chunk_type="table",
                    metadata={
                        "caption": table_result.get("caption", ""),
                        "rows": table_result.get("rows", 0),
                        "columns": table_result.get("columns", 0),
                        "quality": table_result.get("quality", ""),
                        "empty_ratio": table_result.get("empty_ratio", 0.0),
                    },
                )
                chunks.append(chunk)

        elif rtype == "figure":
            fig_result = None
            try:
                fig_result = extract_figure(filepath, page_num, bbox)
            except Exception as exc:
                logger.warning("Figure extraction failed page %d: %s", page_num, exc)

            if fig_result and fig_result.get("content"):
                figure_bboxes.append(bbox)
                chunk = create_atomic_chunk(
                    content=fig_result["content"],
                    doc_id=doc_id,
                    page_num=page_num,
                    chunk_type="figure",
                    image_path=fig_result.get("image_path"),
                    metadata={
                        "has_vision": fig_result.get("has_vision", False),
                        "has_ocr": fig_result.get("has_ocr", False),
                    },
                )
                chunks.append(chunk)

    # ── Second pass: text and scanned pages ───────────────────────────────────
    # Exclude table and figure bboxes so we don't duplicate their content.

    exclude_bboxes = table_bboxes + figure_bboxes

    # Track whether we've already produced text/OCR for this page
    # (the classifier may emit multiple text regions per page)
    text_already_extracted = False
    scanned_already_extracted = False

    for region in regions:
        rtype = region["type"]

        if rtype == "scanned" and not scanned_already_extracted:
            scanned_already_extracted = True
            try:
                ocr_result = extract_scanned_page(filepath, page_num)
                ocr_text = ocr_result.get("text", "")
                if ocr_text.strip():
                    page_chunks = chunk_text(ocr_text, doc_id, page_num)
                    # Tag each chunk with OCR metadata
                    for ch in page_chunks:
                        ch.metadata.update({
                            "ocr": True,
                            "mean_confidence": ocr_result.get("mean_confidence", 0),
                            "low_confidence": ocr_result.get("low_confidence", True),
                        })
                    chunks.extend(page_chunks)
                else:
                    logger.debug("OCR returned empty text for page %d.", page_num)
            except Exception as exc:
                logger.warning("OCR failed on page %d: %s", page_num, exc)

        elif rtype == "text" and not text_already_extracted:
            text_already_extracted = True
            try:
                text_result = extract_text(
                    filepath, page_num, exclude_regions=exclude_bboxes
                )
                text_content = text_result.get("text", "")
                if text_content.strip():
                    page_chunks = chunk_text(text_content, doc_id, page_num)
                    chunks.extend(page_chunks)
                else:
                    logger.debug("No text extracted from page %d.", page_num)
            except Exception as exc:
                logger.warning("Text extraction failed on page %d: %s", page_num, exc)

    return chunks


def ingest_document(
    filepath: str,
    collection_name: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> dict:
    """
    Full ingestion pipeline: extract → chunk → embed → store in Qdrant.

    Parameters
    ----------
    filepath : str
        Path to the PDF to ingest.
    collection_name : str, optional
        Qdrant collection name.  Defaults to ``config.COLLECTION_NAME``.
    doc_id : str, optional
        Document identifier.  Defaults to the filename.

    Returns
    -------
    dict
        {
            "doc_id"         : str,
            "filepath"       : str,
            "collection"     : str,
            "chunks_total"   : int,
            "chunks_stored"  : int,
        }
    """
    from . import config as cfg
    from .embeddings.embedder import embedder_instance
    from .storage.qdrant_store import QdrantStore
    import numpy as np

    collection_name = collection_name or cfg.COLLECTION_NAME
    doc_id = doc_id or os.path.basename(filepath)

    # Step 1 & 2: extract + chunk
    chunks = process_document(filepath, doc_id)

    if not chunks:
        logger.warning("No chunks produced for '%s'. Nothing stored.", filepath)
        return {
            "doc_id": doc_id,
            "filepath": filepath,
            "collection": collection_name,
            "chunks_total": 0,
            "chunks_stored": 0,
        }

    # Step 3: embed
    logger.info("Embedding %d chunks …", len(chunks))
    texts = [chunk.content for chunk in chunks]
    vectors: np.ndarray = embedder_instance.embed_batch(texts)

    # Step 4: store
    store = QdrantStore(collection_name)
    store.upsert_chunks(chunks, vectors)
    logger.info(
        "Stored %d chunks in collection '%s'.", len(chunks), collection_name
    )

    return {
        "doc_id": doc_id,
        "filepath": filepath,
        "collection": collection_name,
        "chunks_total": len(chunks),
        "chunks_stored": len(chunks),
    }
