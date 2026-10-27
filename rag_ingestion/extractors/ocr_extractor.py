import logging
import io
import statistics
from typing import Dict, Any

import fitz
import pytesseract
from PIL import Image

from .. import config

logger = logging.getLogger(__name__)


def extract_scanned_page(
    filepath: str,
    page_num: int,
    dpi: int = None,
    lang: str = None,
    psm: int = None,
) -> Dict[str, Any]:
    """
    OCR a scanned PDF page using Tesseract.

    Parameters
    ----------
    filepath : str
        Path to the PDF file.
    page_num : int
        0-based page index.
    dpi : int, optional
        Render resolution.  Defaults to ``config.OCR_DPI`` (300).
    lang : str, optional
        Tesseract language string (e.g. ``"eng+fra"``).
        Defaults to ``config.OCR_LANG``.
    psm : int, optional
        Tesseract page segmentation mode.
        Defaults to ``config.OCR_PSM`` (3 = auto).

    Returns
    -------
    dict
        {
            "text"             : str,
            "page_num"         : int,
            "source"           : str,
            "ocr"              : True,
            "mean_confidence"  : float,   # ignores Tesseract -1 (N/A) entries
            "median_confidence": float,
            "low_confidence"   : bool,
            "word_count"       : int,
        }
    """

    dpi = dpi if dpi is not None else config.OCR_DPI
    lang = lang if lang is not None else config.OCR_LANG
    psm = psm if psm is not None else config.OCR_PSM

    # ── Validate page number ─────────────────────────────────────────────────
    doc = fitz.open(filepath)
    try:
        if page_num < 0 or page_num >= len(doc):
            raise ValueError(
                f"Invalid page number {page_num}. "
                f"Document has {len(doc)} pages."
            )

        page = doc[page_num]

        # Render at the configured DPI
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        img_data = pix.tobytes("png")

    finally:
        doc.close()

    img = Image.open(io.BytesIO(img_data))

    # ── Tesseract OCR ────────────────────────────────────────────────────────
    tess_config = f"--psm {psm}"
    logger.debug(
        "Running Tesseract on page %d (dpi=%d lang=%s psm=%d)",
        page_num, dpi, lang, psm,
    )

    try:
        data = pytesseract.image_to_data(
            img,
            output_type=pytesseract.Output.DICT,
            config=tess_config,
            lang=lang,
        )
    except pytesseract.TesseractError as exc:
        logger.error("Tesseract error on page %d: %s", page_num, exc)
        return _empty_result(filepath, page_num)

    # ── Collect words + valid confidences ────────────────────────────────────
    words: list[str] = []
    valid_confidences: list[float] = []

    for word_text, conf_raw in zip(data["text"], data["conf"]):
        word_text = str(word_text).strip()

        # Parse confidence
        try:
            conf = float(conf_raw)
        except (ValueError, TypeError):
            continue

        # IMPORTANT: Tesseract returns -1 for non-word tokens (spaces, layout).
        # These are NOT real confidence values and must be excluded from
        # statistics, or they will dramatically deflate the mean.
        if conf < 0:
            continue

        if word_text:
            words.append(word_text)
            valid_confidences.append(conf)

    text = " ".join(words).strip()

    # ── Compute confidence statistics ────────────────────────────────────────
    if valid_confidences:
        mean_conf = sum(valid_confidences) / len(valid_confidences)
        median_conf = statistics.median(valid_confidences)
        low_conf = mean_conf < config.MIN_OCR_CONFIDENCE
    else:
        mean_conf = 0.0
        median_conf = 0.0
        low_conf = True

    if not text:
        logger.warning(
            "Tesseract returned no text for page %d (lang=%s).", page_num, lang
        )

    logger.debug(
        "OCR page %d: %d words, mean_conf=%.1f, median_conf=%.1f, low=%s",
        page_num, len(words), mean_conf, median_conf, low_conf,
    )

    return {
        "text": text,
        "page_num": page_num,
        "source": filepath,
        "ocr": True,
        "mean_confidence": round(mean_conf, 2),
        "median_confidence": round(median_conf, 2),
        "low_confidence": low_conf,
        "word_count": len(words),
    }


def _empty_result(filepath: str, page_num: int) -> Dict[str, Any]:
    return {
        "text": "",
        "page_num": page_num,
        "source": filepath,
        "ocr": True,
        "mean_confidence": 0.0,
        "median_confidence": 0.0,
        "low_confidence": True,
        "word_count": 0,
    }