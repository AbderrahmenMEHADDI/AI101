import fitz
import re
from typing import Dict, Any, List, Optional, Tuple

BBox = Tuple[float, float, float, float]


# ─────────────────────────────────────────────────────────────────────────────
# Header / footer detection
# ─────────────────────────────────────────────────────────────────────────────

# Fraction of page height from top/bottom considered the header/footer zone.
_HEADER_ZONE = 0.07   # top 7%
_FOOTER_ZONE = 0.07   # bottom 7%

_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s*)?\d+\s*$", re.IGNORECASE)


def _is_header_or_footer(block_y0: float, block_y1: float, page_height: float) -> bool:
    """
    Return True if a text block is likely a header or footer.

    A block is flagged when its vertical centre lies within the top
    _HEADER_ZONE or bottom _FOOTER_ZONE fraction of the page.
    """
    centre_y = (block_y0 + block_y1) / 2
    relative_y = centre_y / page_height if page_height > 0 else 0.5

    return relative_y <= _HEADER_ZONE or relative_y >= (1.0 - _FOOTER_ZONE)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_centre(bbox: BBox) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2, (y0 + y1) / 2


def _point_in_bbox(cx: float, cy: float, bbox: BBox) -> bool:
    x0, y0, x1, y1 = bbox
    return x0 <= cx <= x1 and y0 <= cy <= y1


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_text(
    filepath: str,
    page_num: int,
    exclude_regions: Optional[List[BBox]] = None,
) -> Dict[str, Any]:
    """
    Extract native PDF text from a single page.

    Parameters
    ----------
    filepath : str
        Path to the PDF file.
    page_num : int
        0-based page index.
    exclude_regions : list of (x0, y0, x1, y1), optional
        Bounding boxes to skip (e.g. table and figure regions).
        Any text block whose centre falls inside an excluded region
        is omitted.  This prevents duplication between the text
        extractor and the table/figure extractors.

    Returns
    -------
    dict
        {
            "text": str,
            "page_num": int,
            "source": str,
            "native_text": True,
            "word_count": int,
        }
    """
    exclude_regions = exclude_regions or []

    doc = fitz.open(filepath)

    try:
        if page_num < 0 or page_num >= len(doc):
            raise ValueError(
                f"Invalid page number {page_num}. "
                f"Document has {len(doc)} pages."
            )

        page = doc[page_num]
        page_height: float = page.rect.height

        # Get text at block level so we can filter per-block
        blocks = page.get_text("blocks")

        kept_blocks: List[str] = []

        for block in blocks:
            # Blocks: (x0, y0, x1, y1, text, block_no, block_type)
            if len(block) < 5:
                continue

            x0, y0, x1, y1 = block[0], block[1], block[2], block[3]
            text = block[4]
            block_type = block[6] if len(block) > 6 else 0

            # Skip image blocks (type 1)
            if block_type == 1:
                continue

            text = text.strip()
            if not text:
                continue

            # ── Skip header / footer ──────────────────────────────────────
            if _is_header_or_footer(y0, y1, page_height):
                # Only skip if it looks like a page number or very short
                if _PAGE_NUMBER_RE.match(text) or len(text) < 40:
                    continue

            # ── Skip blocks whose centre is inside an excluded region ─────
            cx, cy = _bbox_centre((x0, y0, x1, y1))
            inside_excluded = any(
                _point_in_bbox(cx, cy, exc) for exc in exclude_regions
            )
            if inside_excluded:
                continue

            kept_blocks.append(text)

    finally:
        doc.close()

    raw_text = "\n\n".join(kept_blocks)
    cleaned = clean_text(raw_text)

    return {
        "text": cleaned,
        "page_num": page_num,
        "source": filepath,
        "native_text": True,
        "word_count": len(cleaned.split()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving meaningful structure.

    Steps
    -----
    1. Fix line-break hyphenation  e.g. "transfor-\\nмation" → "transformation"
       BUT only when both sides are word characters (avoids merging em-dashes).
    2. Normalise Windows line endings.
    3. Collapse spaces and tabs within a line.
    4. Limit consecutive blank lines to two.
    5. Strip trailing/leading whitespace per line.
    """
    if not text:
        return ""

    # 1. Fix soft hyphenation across a line break
    #    \\w-\\n\\w  →  join the two word parts
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    # 2. Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Collapse spaces and tabs (but not newlines) within a line
    text = re.sub(r"[ \t]+", " ", text)

    # 4. Limit consecutive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Strip per-line leading/trailing spaces
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()