import os
from .pdf_loader import load_pdf


def get_page_count(filepath: str) -> int:
    """
    Detect file type based on extension, route to the correct loader,
    and return the number of pages as an integer.

    This is a routing layer — future formats (DOCX, HTML) can be added here.

    Returns
    -------
    int
        Number of pages in the document.

    Raises
    ------
    ValueError
        If the file extension is unsupported.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        doc_info = load_pdf(filepath)
        return doc_info["page_count"]   # FIXED: return int, not dict
    else:
        raise ValueError(f"Unsupported file format: {ext}")
