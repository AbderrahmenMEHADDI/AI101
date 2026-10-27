import fitz
import os
from typing import Dict, Any


def load_pdf(filepath: str) -> Dict[str, Any]:
    """
    Load and validate a PDF.

    The loader is responsible only for opening the document
    and returning basic document metadata.

    Content extraction is handled later by the pipeline.

    Returns:
        {
            "filepath": str,
            "filename": str,
            "page_count": int,
            "metadata": dict
        }
    """

    # -----------------------------------------------------
    # Validate filepath
    # -----------------------------------------------------

    if not filepath:
        raise ValueError(
            "PDF filepath cannot be empty."
        )

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"PDF not found: {filepath}"
        )

    if not os.path.isfile(filepath):
        raise ValueError(
            f"Path is not a file: {filepath}"
        )

    # -----------------------------------------------------
    # Open PDF
    # -----------------------------------------------------

    try:
        doc = fitz.open(filepath)
    except Exception as e:
        raise ValueError(
            f"Unable to open PDF '{filepath}': {e}"
        ) from e

    try:

        if doc.is_closed:
            raise ValueError(
                "PDF was unexpectedly closed."
            )

        page_count = len(doc)

        # -------------------------------------------------
        # Document metadata
        # -------------------------------------------------

        metadata = doc.metadata or {}

        return {
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "page_count": page_count,
            "metadata": {
                "title": metadata.get("title"),
                "author": metadata.get("author"),
                "subject": metadata.get("subject"),
                "keywords": metadata.get("keywords"),
                "creator": metadata.get("creator"),
                "producer": metadata.get("producer"),
                "format": metadata.get("format"),
            }
        }

    finally:
        doc.close()