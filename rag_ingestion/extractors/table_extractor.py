import pdfplumber
import pandas as pd
import re
from typing import Tuple, Optional, Dict, Any


TABLE_CAPTION_PATTERN = re.compile(
    r"^\s*(Table|Tab\.?)\s+\d+[\s:.\-]",
    re.IGNORECASE
)


def _sanitize_bbox(
    bbox: Tuple[float, float, float, float],
    page_bbox: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Clamp bbox to the page boundaries and validate it is non-degenerate."""
    px0, py0, px1, py1 = page_bbox
    x0, y0, x1, y1 = bbox

    x0 = max(x0, px0)
    y0 = max(y0, py0)
    x1 = min(x1, px1)
    y1 = min(y1, py1)

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Bounding box has zero or negative area after clamping: {(x0, y0, x1, y1)}")

    return (x0, y0, x1, y1)


def extract_table(
    filepath: str,
    page_num: int,
    bbox: Tuple[float, float, float, float]
) -> Optional[Dict[str, Any]]:
    """
    Extract a table from a PDF bounding box.

    Returns structured table information suitable for
    downstream RAG processing.
    """

    with pdfplumber.open(filepath) as doc:

        if page_num < 0 or page_num >= len(doc.pages):
            raise ValueError(
                f"Invalid page number {page_num}. "
                f"Document contains {len(doc.pages)} pages."
            )

        page = doc.pages[page_num]

        # ------------------------------------------------
        # 1. Crop table region
        # ------------------------------------------------

        safe_bbox = _sanitize_bbox(bbox, page.bbox)
        cropped_page = page.within_bbox(safe_bbox)

        # ------------------------------------------------
        # 2. Extract table
        # ------------------------------------------------

        table_data = cropped_page.extract_table()

        if not table_data:
            return None

        # Remove completely empty rows
        table_data = [
            row for row in table_data
            if row and any(
                cell is not None and str(cell).strip()
                for cell in row
            )
        ]

        if not table_data:
            return None

        # ------------------------------------------------
        # 3. Normalize rows
        # ------------------------------------------------

        max_columns = max(
            len(row)
            for row in table_data
        )

        normalized_rows = []

        for row in table_data:

            row = list(row)

            # Pad short rows
            if len(row) < max_columns:
                row.extend(
                    [""] * (max_columns - len(row))
                )

            # Trim long rows
            row = row[:max_columns]

            normalized_rows.append(
                [
                    "" if cell is None else str(cell).strip()
                    for cell in row
                ]
            )

        # ------------------------------------------------
        # 4. Build DataFrame
        # ------------------------------------------------

        header = normalized_rows[0]
        body = normalized_rows[1:]

        # Avoid completely empty column names
        header = [
            value if value else f"Column_{i + 1}"
            for i, value in enumerate(header)
        ]

        df = pd.DataFrame(
            body,
            columns=header
        )

        df = df.fillna("")

        # ------------------------------------------------
        # 5. Basic quality checks
        # ------------------------------------------------

        rows = len(df)
        columns = len(df.columns)

        total_cells = rows * columns

        empty_cells = (
            df.astype(str)
            .apply(
                lambda col: col.str.strip() == ""
            )
            .sum()
            .sum()
        )

        empty_ratio = (
            empty_cells / total_cells
            if total_cells
            else 1.0
        )

        if rows == 0:
            quality = "poor"
        elif empty_ratio > 0.5:
            quality = "suspicious"
        else:
            quality = "good"

        # ------------------------------------------------
        # 6. Convert to Markdown
        # ------------------------------------------------

        markdown_table = df.to_markdown(
            index=False
        )

        # ------------------------------------------------
        # 7. Caption
        # ------------------------------------------------

        caption = find_table_caption(
            page,
            bbox
        )

        # ------------------------------------------------
        # 8. Final representation
        # ------------------------------------------------

        if caption:
            content = (
                f"{caption}\n\n"
                f"{markdown_table}"
            )
        else:
            content = markdown_table

        return {
            "type": "table",
            "page_num": page_num,
            "text": content,
            "markdown": markdown_table,
            "caption": caption,
            "rows": rows,
            "columns": columns,
            "empty_ratio": round(
                empty_ratio,
                3
            ),
            "quality": quality,
            "extraction_method": "pdfplumber",
        }


def find_table_caption(
    page,
    table_bbox
) -> str:
    """
    Find the nearest table caption above the table.
    """

    table_top = table_bbox[1]

    words = page.extract_words()

    if not words:
        return ""

    candidate_lines = {}

    for word in words:

        text = word.get("text", "").strip()

        if not text:
            continue

        if not TABLE_CAPTION_PATTERN.match(text):
            continue

        top = word["top"]

        # Caption must be above table
        if top >= table_top:
            continue

        line_key = round(top, 1)

        candidate_lines.setdefault(
            line_key,
            []
        ).append(word)

    if not candidate_lines:
        return ""

    # Find closest caption above table
    closest_top = max(
        candidate_lines.keys()
    )

    words_in_line = sorted(
        candidate_lines[closest_top],
        key=lambda w: w["x0"]
    )

    return " ".join(
        word["text"]
        for word in words_in_line
    )