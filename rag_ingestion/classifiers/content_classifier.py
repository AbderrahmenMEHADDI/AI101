import fitz  # PyMuPDF
import pdfplumber

from typing import List, Dict, Any, Tuple


# ---------------------------------------------------------
# Types
# ---------------------------------------------------------

BBox = Tuple[float, float, float, float]


# ---------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------

def normalize_bbox(bbox: BBox) -> BBox:
    """
    Ensure bbox is ordered as:
        (x0, y0, x1, y1)
    """

    x0, y0, x1, y1 = bbox

    return (
        min(x0, x1),
        min(y0, y1),
        max(x0, x1),
        max(y0, y1),
    )


def bbox_area(bbox: BBox) -> float:
    """
    Calculate bounding-box area.
    """

    x0, y0, x1, y1 = normalize_bbox(bbox)

    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)

    return width * height


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    """
    Return the center point of a bounding box.
    """

    x0, y0, x1, y1 = normalize_bbox(bbox)

    return (
        (x0 + x1) / 2,
        (y0 + y1) / 2,
    )


def point_in_bbox(
    point: Tuple[float, float],
    bbox: BBox
) -> bool:
    """
    Check whether a point lies inside a bounding box.
    """

    x, y = point

    x0, y0, x1, y1 = normalize_bbox(bbox)

    return (
        x0 <= x <= x1
        and
        y0 <= y <= y1
    )


def bbox_intersection_area(
    bbox_a: BBox,
    bbox_b: BBox
) -> float:
    """
    Calculate intersection area between two bounding boxes.
    """

    ax0, ay0, ax1, ay1 = normalize_bbox(bbox_a)
    bx0, by0, bx1, by1 = normalize_bbox(bbox_b)

    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)

    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)

    if x1 <= x0 or y1 <= y0:
        return 0.0

    return (x1 - x0) * (y1 - y0)


def overlap_ratio(
    bbox_a: BBox,
    bbox_b: BBox
) -> float:
    """
    Intersection over the area of bbox_a.

    Useful for determining whether an image is
    mostly contained inside a table.
    """

    area_a = bbox_area(bbox_a)

    if area_a == 0:
        return 0.0

    intersection = bbox_intersection_area(
        bbox_a,
        bbox_b
    )

    return intersection / area_a


# ---------------------------------------------------------
# Deduplication
# ---------------------------------------------------------

def is_duplicate_bbox(
    bbox: BBox,
    existing: List[BBox],
    threshold: float = 0.90
) -> bool:
    """
    Determine whether a bounding box is essentially
    identical to an existing bounding box.
    """

    for existing_bbox in existing:

        overlap = overlap_ratio(
            bbox,
            existing_bbox
        )

        reverse_overlap = overlap_ratio(
            existing_bbox,
            bbox
        )

        if (
            overlap >= threshold
            and
            reverse_overlap >= threshold
        ):
            return True

    return False


# ---------------------------------------------------------
# Scanned-page detection
# ---------------------------------------------------------

def calculate_image_coverage(
    page: fitz.Page
) -> float:
    """
    Estimate how much of the page is covered by images.

    Returns a value between 0 and 1.
    """

    page_area = bbox_area(
        (
            page.rect.x0,
            page.rect.y0,
            page.rect.x1,
            page.rect.y1
        )
    )

    if page_area <= 0:
        return 0.0

    image_area = 0.0

    seen_bboxes = []

    for image in page.get_images(full=True):

        xref = image[0]

        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue

        for rect in rects:

            bbox = (
                rect.x0,
                rect.y0,
                rect.x1,
                rect.y1
            )

            if is_duplicate_bbox(
                bbox,
                seen_bboxes
            ):
                continue

            seen_bboxes.append(bbox)

            image_area += bbox_area(bbox)

    # Prevent values > 100% because images may overlap.
    return min(
        image_area / page_area,
        1.0
    )


def is_scanned_page(
    page: fitz.Page,
    native_text: str,
    image_coverage_threshold: float = 0.75,
    max_native_text_chars: int = 100
) -> bool:
    """
    Determine whether a page is likely a scanned page.

    A page is considered scanned when:
      - it has very little native text
      - and images cover most of the page
    """

    if not native_text:
        native_text = ""

    image_coverage = calculate_image_coverage(page)

    return (
        len(native_text.strip())
        <= max_native_text_chars
        and
        image_coverage
        >= image_coverage_threshold
    )


# ---------------------------------------------------------
# Main classifier
# ---------------------------------------------------------

def classify_page_content(
    filepath: str,
    page_num: int
) -> List[Dict[str, Any]]:
    """
    Classify content regions on a PDF page.

    Possible region types:

        text
        table
        figure
        scanned

    Each region contains:

        {
            "type": str,
            "bbox": [x0, y0, x1, y1],
            "page_num": int,
            "confidence": float,
            "metadata": {...}
        }

    Important:
    Text regions exclude areas already classified
    as tables or figures.
    """

    regions: List[Dict[str, Any]] = []

    # -----------------------------------------------------
    # Open PDF with PyMuPDF
    # -----------------------------------------------------

    doc = fitz.open(filepath)

    try:

        if page_num < 0 or page_num >= len(doc):
            raise ValueError(
                f"Invalid page number {page_num}. "
                f"Document contains {len(doc)} pages."
            )

        page = doc[page_num]

        page_bbox: BBox = (
            page.rect.x0,
            page.rect.y0,
            page.rect.x1,
            page.rect.y1
        )

        # -------------------------------------------------
        # Native text
        # -------------------------------------------------

        native_text = page.get_text(
            "text"
        ).strip()

        # -------------------------------------------------
        # SCANNED PAGE
        # -------------------------------------------------

        if is_scanned_page(
            page,
            native_text
        ):

            regions.append({
                "type": "scanned",
                "bbox": list(page_bbox),
                "page_num": page_num,
                "confidence": 0.90,
                "metadata": {
                    "reason": "large_image_coverage_low_native_text",
                    "image_coverage": round(
                        calculate_image_coverage(page),
                        3
                    ),
                    "native_text_chars": len(
                        native_text
                    ),
                }
            })

            return regions

        # -------------------------------------------------
        # TABLE DETECTION
        # -------------------------------------------------

        table_bboxes: List[BBox] = []

        try:

            with pdfplumber.open(filepath) as pdf:

                plumber_page = pdf.pages[page_num]

                tables = plumber_page.find_tables()

                for table in tables:

                    # pdfplumber gives:
                    # (x0, top, x1, bottom)
                    #
                    # For normal PDF pages this corresponds
                    # to the same top-left coordinate system
                    # we use here.

                    raw_bbox = table.bbox

                    bbox: BBox = normalize_bbox(
                        (
                            raw_bbox[0],
                            raw_bbox[1],
                            raw_bbox[2],
                            raw_bbox[3]
                        )
                    )

                    # Ignore tiny accidental detections
                    if bbox_area(bbox) < 500:
                        continue

                    # Avoid duplicate table detections
                    if is_duplicate_bbox(
                        bbox,
                        table_bboxes
                    ):
                        continue

                    table_bboxes.append(bbox)

                    regions.append({
                        "type": "table",
                        "bbox": list(bbox),
                        "page_num": page_num,
                        "confidence": 0.85,
                        "metadata": {
                            "detector": "pdfplumber"
                        }
                    })

        except Exception as e:

            # Don't destroy the whole pipeline if table
            # detection fails.
            print(
                f"[WARN] Table detection failed "
                f"on page {page_num}: {e}"
            )

        # -------------------------------------------------
        # FIGURE / IMAGE DETECTION
        # -------------------------------------------------

        figure_bboxes: List[BBox] = []

        image_list = page.get_images(
            full=True
        )

        for image in image_list:

            xref = image[0]

            try:
                rects = page.get_image_rects(
                    xref
                )
            except Exception:
                continue

            for rect in rects:

                bbox: BBox = (
                    rect.x0,
                    rect.y0,
                    rect.x1,
                    rect.y1
                )

                bbox = normalize_bbox(bbox)

                # Ignore tiny images such as:
                # logos, icons, bullets, tracking pixels, etc.
                if bbox_area(bbox) < 500:
                    continue

                # -------------------------------------------------
                # Skip duplicate image placements
                # -------------------------------------------------

                if is_duplicate_bbox(
                    bbox,
                    figure_bboxes
                ):
                    continue

                # -------------------------------------------------
                # Check whether image belongs to a table
                # -------------------------------------------------

                inside_table = False

                for table_bbox in table_bboxes:

                    image_overlap = overlap_ratio(
                        bbox,
                        table_bbox
                    )

                    if image_overlap >= 0.50:
                        inside_table = True
                        break

                if inside_table:
                    continue

                figure_bboxes.append(
                    bbox
                )

                regions.append({
                    "type": "figure",
                    "bbox": list(bbox),
                    "page_num": page_num,
                    "confidence": 0.80,
                    "metadata": {
                        "detector": "pymupdf",
                        "xref": xref
                    }
                })

        # -----------------------------------------------------
        # TEXT REGION DETECTION
        # -----------------------------------------------------

        # IMPORTANT:
        #
        # Instead of treating the whole page as text,
        # inspect individual text blocks.
        #
        # This prevents table/figure content from being
        # automatically duplicated in the native text output.

        text_blocks = page.get_text(
            "blocks"
        )

        excluded_regions = (
            table_bboxes
            +
            figure_bboxes
        )

        for block in text_blocks:

            if len(block) < 5:
                continue

            x0, y0, x1, y1 = block[:4]

            block_text = block[4].strip()

            if not block_text:
                continue

            block_bbox: BBox = (
                x0,
                y0,
                x1,
                y1
            )

            # ---------------------------------------------
            # Ignore text blocks inside tables/figures
            # ---------------------------------------------

            excluded = False

            for excluded_bbox in excluded_regions:

                center = bbox_center(
                    block_bbox
                )

                if point_in_bbox(
                    center,
                    excluded_bbox
                ):
                    excluded = True
                    break

            if excluded:
                continue

            regions.append({
                "type": "text",
                "bbox": list(block_bbox),
                "page_num": page_num,
                "confidence": 1.0,
                "metadata": {
                    "detector": "pymupdf",
                    "text": block_text
                }
            })

        # -----------------------------------------------------
        # Sort regions by reading order
        # -----------------------------------------------------

        regions.sort(
            key=lambda region: (
                region["bbox"][1],
                region["bbox"][0]
            )
        )

        return regions

    finally:
        doc.close()