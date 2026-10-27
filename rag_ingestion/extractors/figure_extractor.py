"""
Figure Extractor
================
Crops a figure region from a PDF page, saves it as a PNG,
runs Tesseract OCR to extract any embedded text, and calls
a local vision-language model (via Ollama) to generate a
semantic description.

Vision captioning is modular: change ``describe_image()``
to swap in a different backend without touching the rest of
the pipeline.
"""

import logging
import os
import uuid
from typing import Dict, Any, Tuple

import fitz
import pytesseract
from PIL import Image

from .. import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Vision captioner — modular, swappable backend
# ─────────────────────────────────────────────────────────────────────────────

_VISION_PROMPT = (
    "You are analyzing a figure extracted from a document for a RAG system.\n\n"
    "Describe the figure accurately and concisely.\n\n"
    "Include:\n"
    "- Type of figure (chart, diagram, photograph, flowchart, screenshot, etc.)\n"
    "- Main subject\n"
    "- Important components\n"
    "- Relationships between components\n"
    "- Labels and annotations\n"
    "- Axes, legends, and trends if present\n"
    "- Important numerical information\n"
    "- Any conclusions that can safely be inferred\n\n"
    "Do not invent information that is not visible."
)


def describe_image(image_path: str) -> str:
    """
    Generate a semantic description of an image using a local vision-language
    model served by Ollama.

    Parameters
    ----------
    image_path : str
        Absolute path to the saved image file.

    Returns
    -------
    str
        A textual description, or an empty string if the model is unavailable.

    Notes
    -----
    If Ollama is not running or the model is not pulled, this function logs a
    clear error with setup instructions and returns an empty string.
    The caller combines this with OCR text so the chunk is never empty.
    """
    if not os.path.exists(image_path):
        logger.error("Image not found for captioning: %s", image_path)
        return ""

    vision_model = config.VISION_MODEL

    try:
        import ollama  # noqa: PLC0415 — optional dependency
    except ImportError:
        logger.warning(
            "ollama package not installed. Vision captioning unavailable.\n"
            "Install with: pip install ollama\n"
            "Then pull the model: ollama pull %s",
            vision_model,
        )
        return ""

    try:
        response = ollama.chat(
            model=vision_model,
            messages=[
                {
                    "role": "user",
                    "content": _VISION_PROMPT,
                    "images": [image_path],
                }
            ],
        )
        description = response["message"]["content"].strip()
        logger.debug(
            "Vision model '%s' described image: %s chars", vision_model, len(description)
        )
        return description

    except Exception as exc:
        # Provide actionable guidance on how to fix this
        logger.warning(
            "Vision captioning failed for '%s': %s\n"
            "To enable vision captioning:\n"
            "  1. Install Ollama: https://ollama.com/download\n"
            "  2. Pull the vision model: ollama pull %s\n"
            "  3. Ensure Ollama is running: ollama serve\n"
            "  4. Set VISION_MODEL in .env if you want a different model.\n"
            "Falling back to OCR-only for this figure.",
            image_path, exc, vision_model,
        )
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Bounding box helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_bbox(
    bbox: Tuple[float, float, float, float],
    page: fitz.Page,
) -> Tuple[float, float, float, float]:
    """
    Clamp bbox to the page boundaries and validate it is non-degenerate.
    """
    x0, y0, x1, y1 = bbox
    rect = page.rect

    x0 = max(x0, rect.x0)
    y0 = max(y0, rect.y0)
    x1 = min(x1, rect.x1)
    y1 = min(y1, rect.y1)

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Bounding box has zero or negative area: {(x0, y0, x1, y1)}")

    return (x0, y0, x1, y1)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_figure(
    filepath: str,
    page_num: int,
    bbox: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    """
    Extract a figure from a PDF page and produce a rich description.

    Steps
    -----
    1. Crop the figure region at 200 DPI.
    2. Save as PNG to ``config.FIGURES_DIR``.
    3. Run Tesseract OCR to capture any text in the figure.
    4. Call the vision-language model to describe the figure.
    5. Combine description + OCR into a single textual representation.

    Parameters
    ----------
    filepath : str
        Path to the PDF file.
    page_num : int
        0-based page index.
    bbox : (x0, y0, x1, y1)
        PyMuPDF coordinate bounding box of the figure (top-left origin).

    Returns
    -------
    dict
        {
            "content"    : str,   # combined description + OCR text
            "image_path" : str,   # path to saved PNG
            "page_num"   : int,
            "has_vision" : bool,  # True if vision model produced a description
            "has_ocr"    : bool,  # True if OCR found text
        }
    """
    doc = fitz.open(filepath)
    try:
        if page_num < 0 or page_num >= len(doc):
            raise ValueError(
                f"Invalid page number {page_num}. Document has {len(doc)} pages."
            )

        page = doc[page_num]
        safe_bbox = _sanitize_bbox(bbox, page)
        rect = fitz.Rect(safe_bbox)

        # Render figure region at 200 DPI
        pix = page.get_pixmap(clip=rect, dpi=200, alpha=False)

    finally:
        doc.close()

    # ── Save image ────────────────────────────────────────────────────────────
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    image_filename = f"figure_{uuid.uuid4().hex[:12]}.png"
    image_path = os.path.join(config.FIGURES_DIR, image_filename)
    pix.save(image_path)
    logger.debug("Saved figure to %s", image_path)

    # ── OCR ───────────────────────────────────────────────────────────────────
    img = Image.open(image_path)
    try:
        ocr_text = pytesseract.image_to_string(img).strip()
    except Exception as exc:
        logger.warning("Figure OCR failed: %s", exc)
        ocr_text = ""

    # ── Vision description ────────────────────────────────────────────────────
    description = describe_image(image_path)

    # ── Compose content ───────────────────────────────────────────────────────
    parts: list[str] = []

    if description:
        parts.append(f"Figure Description:\n{description}")
    else:
        # No silent placeholder — be explicit so the LLM knows
        parts.append(
            "Figure Description:\n"
            "[Vision model unavailable — description not generated. "
            "See OCR text below if present.]"
        )

    if ocr_text:
        parts.append(f"Text found in figure:\n{ocr_text}")

    content = "\n\n".join(parts)

    return {
        "content": content,
        "image_path": image_path,
        "page_num": page_num,
        "has_vision": bool(description),
        "has_ocr": bool(ocr_text),
    }