"""
Tests for the OCR extractor.

Covers:
- Confidence calculation excludes -1 Tesseract values
- Empty OCR output handled gracefully
- mean_confidence within valid range
- low_confidence flag set correctly
"""

import pytest
from unittest.mock import patch, MagicMock


def _make_tesseract_data(words_and_confs):
    """
    Build a fake pytesseract image_to_data output dict.

    words_and_confs: list of (word, confidence) tuples.
    -1 confidence simulates Tesseract's non-word tokens.
    """
    texts = []
    confs = []
    for word, conf in words_and_confs:
        texts.append(word)
        confs.append(conf)
    return {"text": texts, "conf": confs}


@pytest.fixture
def mock_fitz_page():
    """Minimal fitz mock so we don't need a real PDF."""
    mock_pix = MagicMock()
    mock_pix.tobytes.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    mock_page = MagicMock()
    mock_page.get_pixmap.return_value = mock_pix

    mock_doc = MagicMock()
    mock_doc.__len__ = MagicMock(return_value=5)
    mock_doc.__getitem__ = MagicMock(return_value=mock_page)

    return mock_doc


def _run_ocr(words_and_confs, mock_doc):
    """Run extract_scanned_page with mocked dependencies."""
    from io import BytesIO
    from PIL import Image

    # Create a tiny valid PNG so PIL.Image.open doesn't complain
    import struct, zlib
    def _minimal_png():
        def chunk(tag, data):
            c = struct.pack(">I", len(data)) + tag + data
            return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        header = b"\x89PNG\r\n\x1a\n"
        ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        idat_data = zlib.compress(b"\x00\xff\xff\xff")
        idat = chunk(b"IDAT", idat_data)
        iend = chunk(b"IEND", b"")
        return header + ihdr + idat + iend

    mock_pix = MagicMock()
    mock_pix.tobytes.return_value = _minimal_png()
    mock_doc.__getitem__.return_value.get_pixmap.return_value = mock_pix

    tess_data = _make_tesseract_data(words_and_confs)

    with patch("fitz.open", return_value=mock_doc), \
         patch("pytesseract.image_to_data", return_value=tess_data), \
         patch("PIL.Image.open") as mock_img_open:

        mock_img_open.return_value = MagicMock()

        from rag_ingestion.extractors.ocr_extractor import extract_scanned_page
        return extract_scanned_page("fake.pdf", 0)


def test_confidence_excludes_negative_one(mock_fitz_page):
    """
    Tesseract returns -1 for non-word tokens.
    These must NOT be included in confidence statistics.
    """
    words_and_confs = [
        ("",       -1),   # non-word token — must be excluded
        ("Hello",  90.0),
        ("",       -1),   # another non-word token
        ("world",  80.0),
    ]
    result = _run_ocr(words_and_confs, mock_fitz_page)

    # Mean should be (90 + 80) / 2 = 85, NOT (−1 + 90 − 1 + 80) / 4 = 42
    assert result["mean_confidence"] == pytest.approx(85.0, abs=0.1)
    assert result["word_count"] == 2
    assert result["text"] == "Hello world"


def test_empty_ocr_output(mock_fitz_page):
    """When OCR returns no words, result should be gracefully empty."""
    words_and_confs = [("", -1), ("", -1)]
    result = _run_ocr(words_and_confs, mock_fitz_page)

    assert result["text"] == ""
    assert result["word_count"] == 0
    assert result["mean_confidence"] == 0.0
    assert result["low_confidence"] is True


def test_low_confidence_flag(mock_fitz_page):
    """low_confidence should be True when mean < MIN_OCR_CONFIDENCE."""
    from rag_ingestion import config
    low = config.MIN_OCR_CONFIDENCE - 10  # definitely below threshold
    words_and_confs = [("text1", low), ("text2", low)]
    result = _run_ocr(words_and_confs, mock_fitz_page)
    assert result["low_confidence"] is True


def test_high_confidence_not_flagged(mock_fitz_page):
    """low_confidence should be False when mean >= MIN_OCR_CONFIDENCE."""
    from rag_ingestion import config
    high = config.MIN_OCR_CONFIDENCE + 20
    words_and_confs = [("text1", high), ("text2", high)]
    result = _run_ocr(words_and_confs, mock_fitz_page)
    assert result["low_confidence"] is False


def test_ocr_metadata_fields(mock_fitz_page):
    words_and_confs = [("Test", 75.0), ("word", 85.0)]
    result = _run_ocr(words_and_confs, mock_fitz_page)

    required = {"text", "page_num", "source", "ocr", "mean_confidence",
                "median_confidence", "low_confidence", "word_count"}
    assert required.issubset(result.keys())
    assert result["ocr"] is True
    assert result["page_num"] == 0
