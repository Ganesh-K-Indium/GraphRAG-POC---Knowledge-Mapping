"""
PDF document parser using PyMuPDF (fitz).

Implements :class:`core.interfaces.IParser` for `.pdf` files.

Two-pass extraction:
1. Native text via ``page.get_text()`` (fast, lossless for digital PDFs).
2. OCR fallback via Tesseract when a page yields < 20 characters — handles
   fully scanned PDFs like government RFPs that contain only image layers.

OCR quality filters:
- Pages where > 40 % of characters are non-ASCII are skipped (handles mixed
  Korean/CJK pages that tesseract renders as garbage).
- Pages where the OCR result is still < 20 chars after stripping are dropped.

Table extraction:
- Uses ``page.find_tables()`` (PyMuPDF 1.23+) to extract native PDF tables
  as structured markdown text appended to page content so tables are not lost
  or garbled during text extraction.

Image extraction:
- Scans pages for embedded images via ``page.get_images(full=True)``.
- Filters out decorative images (< 50px in any dimension).
- Enhances images for better OCR (contrast, sharpness, brightness).
- Extracts spatial proximity context (text blocks near the image).
- Runs Tesseract OCR on extracted images for exact text/numbers.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from pathlib import Path
from typing import BinaryIO

import fitz  # PyMuPDF

from core.exceptions import ParseError
from core.interfaces import IParser
from core.models import DocumentType, ImageContext, ParsedDocument

logger = logging.getLogger(__name__)

# Lazy-import OCR deps so the parser works even if they are absent (the
# native-text path still functions for digital PDFs).
_TESSERACT_AVAILABLE: bool | None = None  # None = not yet checked


def _check_tesseract() -> bool:
    global _TESSERACT_AVAILABLE
    if _TESSERACT_AVAILABLE is None:
        try:
            import pytesseract  # noqa: F401
            from PIL import Image  # noqa: F401
            _TESSERACT_AVAILABLE = True
        except ImportError:
            _TESSERACT_AVAILABLE = False
            logger.warning(
                "pytesseract / Pillow not installed — OCR fallback disabled. "
                "Install with: pip install pytesseract Pillow"
            )
    return _TESSERACT_AVAILABLE  # type: ignore[return-value]


def _non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / len(text)


def _ocr_page(page: fitz.Page, dpi: int = 150) -> str:
    """Render *page* to a grayscale image and OCR it with Tesseract.

    Grayscale at 150 DPI uses ~6× less memory than RGB at 200 DPI while
    keeping text quality sufficient for standard RFP/contract documents.
    """
    import pytesseract
    from PIL import Image

    scale = dpi / 72
    mat = fitz.Matrix(scale, scale)
    # Grayscale pixmap: 1 byte per pixel vs 3 for RGB — faster and lighter
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    text: str = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
    return text


# ---------------------------------------------------------------------------
# Table extraction helper
# ---------------------------------------------------------------------------


def _extract_tables_as_markdown(page: fitz.Page) -> str:
    """
    Extract tables from a PDF page and convert to markdown format.

    Uses PyMuPDF's built-in table finder (available from fitz >= 1.23.0).
    Returns an empty string if no tables are found or if the feature is
    unavailable.
    """
    try:
        tables = page.find_tables()
        if not tables or len(tables.tables) == 0:
            return ""

        markdown_parts: list[str] = []
        for table_idx, table in enumerate(tables.tables):
            extracted = table.extract()
            if not extracted:
                continue

            rows: list[str] = []
            for row_idx, row in enumerate(extracted):
                # Replace None cells with empty strings
                cells = [str(cell).strip() if cell else "" for cell in row]
                row_str = "| " + " | ".join(cells) + " |"
                rows.append(row_str)

                # Add header separator after first row
                if row_idx == 0:
                    sep = "| " + " | ".join(["---"] * len(cells)) + " |"
                    rows.append(sep)

            if rows:
                markdown_parts.append(
                    f"\n[TABLE {table_idx + 1}]\n" + "\n".join(rows)
                )

        return "\n".join(markdown_parts)
    except AttributeError:
        # find_tables() not available in this version of PyMuPDF
        logger.debug("page.find_tables() not available — skipping table extraction")
        return ""
    except Exception as exc:
        logger.debug("Table extraction failed on page: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Image extraction helpers
# ---------------------------------------------------------------------------


def _enhance_image(img: "Image.Image") -> "Image.Image":
    """
    Enhance an image for better OCR accuracy.

    Applies contrast (1.4x), sharpness (1.3x), and brightness (1.08x)
    adjustments, and resizes if too small (< 512px) or too large (> 2048px).
    """
    from PIL import ImageEnhance

    # Resize if needed
    w, h = img.size
    if max(w, h) < 512:
        scale = 512 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))
    elif max(w, h) > 2048:
        scale = 2048 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))

    # Enhance for OCR
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    img = ImageEnhance.Brightness(img).enhance(1.08)
    return img


def _get_spatial_context(page: fitz.Page, image_rect: fitz.Rect, max_words: int = 150) -> str:
    """
    Extract text blocks in spatial proximity to an image bounding box.

    Scans all text blocks on the page and returns those within 400px
    vertically of the image rect, trimmed to *max_words*.
    """
    VERTICAL_DISTANCE = 400
    try:
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
        nearby: list[str] = []

        for block in blocks:
            if block[6] != 0:  # skip non-text blocks (images)
                continue
            block_y0 = block[1]
            block_y1 = block[3]
            block_text = block[4].strip()

            if not block_text:
                continue

            # Check vertical proximity
            vertical_dist = min(
                abs(block_y0 - image_rect.y1),  # block is below image
                abs(image_rect.y0 - block_y1),  # block is above image
            )

            if vertical_dist <= VERTICAL_DISTANCE:
                nearby.append(block_text)

        combined = " ".join(nearby)
        words = combined.split()
        if len(words) > max_words:
            words = words[:max_words]
        return " ".join(words)
    except Exception as exc:
        logger.debug("Spatial context extraction failed: %s", exc)
        return ""


def _extract_images_from_page(
    page: fitz.Page,
    pdf_doc: fitz.Document,
    page_num: int,
    filename: str,
    min_dimension: int = 50,
) -> list[ImageContext]:
    """
    Extract, filter, and enhance images from a single PDF page.

    Skips decorative images (where either dimension < *min_dimension*).
    Returns a list of :class:`ImageContext` objects with OCR text and
    spatial context populated.
    """
    if not _check_tesseract():
        return []

    import pytesseract
    from PIL import Image as PILImage

    image_contexts: list[ImageContext] = []

    try:
        images = page.get_images(full=True)
    except Exception as exc:
        logger.debug("get_images() failed on page %d: %s", page_num, exc)
        return []

    for img_info in images:
        xref = img_info[0]

        try:
            base_image = pdf_doc.extract_image(xref)
            if not base_image or "image" not in base_image:
                continue

            raw_bytes = base_image["image"]
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)

            # Skip tiny decorative images
            if width < min_dimension or height < min_dimension:
                continue

            # Open, convert to RGB, enhance
            img = PILImage.open(io.BytesIO(raw_bytes)).convert("RGB")
            enhanced = _enhance_image(img)

            # OCR the enhanced image
            try:
                ocr_text = pytesseract.image_to_string(enhanced, config="--oem 3 --psm 6")
            except Exception as ocr_exc:
                logger.debug("Image OCR failed on page %d: %s", page_num, ocr_exc)
                ocr_text = ""

            # Get spatial context from surrounding text blocks
            # Build an approximate image rect from the page's image list
            try:
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    image_rect = img_rects[0]
                else:
                    # Fallback: use full page rect
                    image_rect = page.rect
            except Exception:
                image_rect = page.rect

            surrounding_text = _get_spatial_context(page, image_rect)

            # Compute hash of raw image bytes for deduplication
            image_hash = hashlib.sha256(raw_bytes).hexdigest()

            # Convert enhanced image to PNG bytes for downstream Vision API
            buf = io.BytesIO()
            enhanced.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            image_contexts.append(
                ImageContext(
                    image_bytes=png_bytes,
                    page_number=page_num,
                    ocr_text=ocr_text.strip(),
                    surrounding_text=surrounding_text,
                    image_hash=image_hash,
                    source_filename=filename,
                )
            )

        except Exception as exc:
            logger.debug(
                "Failed to extract image xref=%d on page %d: %s",
                xref, page_num, exc,
            )
            continue

    return image_contexts


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


class PDFParser(IParser):
    """Parse PDF files into :class:`~core.models.ParsedDocument` objects.

    Parameters
    ----------
    ocr_dpi:
        Resolution used when rendering scanned pages for OCR (default 200 DPI).
        Higher values improve accuracy at the cost of speed.
    non_ascii_threshold:
        Pages where the fraction of non-ASCII characters exceeds this value
        are discarded (catches CJK / garbled OCR output).  Default 0.40.
    """

    def __init__(
        self,
        ocr_dpi: int = 150,
        non_ascii_threshold: float = 0.40,
    ) -> None:
        self.ocr_dpi = ocr_dpi
        self.non_ascii_threshold = non_ascii_threshold

    # ------------------------------------------------------------------
    # IParser contract
    # ------------------------------------------------------------------

    def supports(self, filename: str) -> bool:
        return Path(filename).suffix.lower() == ".pdf"

    def parse(
        self,
        file: BinaryIO,
        filename: str,
        progress_cb=None,
    ) -> ParsedDocument:
        """Extract text from every page, falling back to OCR for scanned pages.

        Table data is extracted via ``page.find_tables()`` and appended as
        markdown to the page text so that tabular data is not lost.

        Parameters
        ----------
        file:
            Open binary stream of the PDF, positioned at byte 0.
        filename:
            Original filename; used to derive the document type and ID.

        Returns
        -------
        ParsedDocument
            One entry in ``pages`` per page that yields usable text.

        Raises
        ------
        core.exceptions.ParseError
            Wraps any underlying :mod:`fitz` or I/O error.
        """
        try:
            data: bytes = file.read()
            pdf_doc = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise ParseError(f"Failed to open PDF '{filename}': {exc}") from exc

        ocr_available = _check_tesseract()
        total_pages = pdf_doc.page_count
        pages: list[str] = []
        ocr_count = 0
        skipped_count = 0

        if progress_cb:
            mode = "OCR" if ocr_available else "text"
            progress_cb(f"  Scanning {total_pages} pages ({mode})…")

        for page_num, page in enumerate(pdf_doc, start=1):
            # ── Pass 1: native text ────────────────────────────────────
            text: str = page.get_text("text")

            # ── Pass 1b: table extraction ──────────────────────────────
            table_md = _extract_tables_as_markdown(page)
            if table_md:
                text = text + "\n\n" + table_md

            # ── Pass 2: OCR fallback for image-only pages ──────────────
            if len(text.strip()) < 20:
                if not ocr_available:
                    continue  # can't OCR, skip blank page
                try:
                    if progress_cb:
                        progress_cb(f"  OCR page {page_num}/{total_pages}…")
                    text = _ocr_page(page, dpi=self.ocr_dpi)
                    ocr_count += 1
                except Exception as exc:
                    logger.warning(
                        "OCR failed for page %d of '%s': %s", page_num, filename, exc
                    )
                    continue

            # ── Quality filters ────────────────────────────────────────
            stripped = text.strip()
            if len(stripped) < 20:
                skipped_count += 1
                continue

            if _non_ascii_ratio(stripped) > self.non_ascii_threshold:
                logger.debug(
                    "Skipping page %d of '%s' (high non-ASCII ratio — likely CJK/form)",
                    page_num,
                    filename,
                )
                skipped_count += 1
                continue

            pages.append(stripped)

        if ocr_count:
            logger.info(
                "Parsed '%s': %d pages (%d via OCR, %d skipped)",
                filename, len(pages), ocr_count, skipped_count,
            )
            if progress_cb:
                progress_cb(
                    f"  OCR complete — {len(pages)} usable pages "
                    f"({ocr_count} OCR'd, {skipped_count} skipped)"
                )
        else:
            logger.info(
                "Parsed '%s': %d pages (%d skipped)",
                filename, len(pages), skipped_count,
            )

        doc_type = _infer_document_type(filename)
        doc_id = _build_document_id(filename)

        return ParsedDocument(
            id=doc_id,
            name=filename,
            type=doc_type,
            pages=pages,
            total_pages=len(pages),
        )

    def extract_images(self, file: BinaryIO, filename: str) -> list[ImageContext]:
        """
        Extract, preprocess, and OCR all meaningful images from the PDF.

        This is a *separate* method from :meth:`parse` so that the
        :class:`IParser` interface is not changed.  Callers that want
        image extraction should call this after ``parse()``.

        Parameters
        ----------
        file:
            Open binary stream of the PDF, positioned at byte 0.
        filename:
            Original filename.

        Returns
        -------
        list[ImageContext]
            One entry per non-decorative image found in the PDF.
        """
        try:
            data: bytes = file.read()
            pdf_doc = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            logger.warning("Failed to open PDF for image extraction '%s': %s", filename, exc)
            return []

        all_images: list[ImageContext] = []
        seen_hashes: set[str] = set()

        for page_num, page in enumerate(pdf_doc, start=1):
            page_images = _extract_images_from_page(
                page, pdf_doc, page_num, filename
            )
            for img_ctx in page_images:
                # Deduplicate images within the same document
                if img_ctx.image_hash in seen_hashes:
                    continue
                seen_hashes.add(img_ctx.image_hash)
                all_images.append(img_ctx)

        logger.info(
            "Extracted %d images from '%s' (%d unique after dedup)",
            len(all_images) + len(seen_hashes) - len(all_images),
            filename,
            len(all_images),
        )
        return all_images


# ---------------------------------------------------------------------------
# Shared helpers (used by docx_parser as well via import)
# ---------------------------------------------------------------------------

def _infer_document_type(filename: str) -> DocumentType:
    name_lower = Path(filename).stem.lower()

    rfp_keywords = ("rfp", "rfx", "tender")
    risk_keywords = ("risk", "rmc", "register")
    contract_keywords = ("contract", "offer", "agreement", "purchase")

    if any(kw in name_lower for kw in rfp_keywords):
        return DocumentType.RFP
    if any(kw in name_lower for kw in risk_keywords):
        return DocumentType.RISK_SHEET
    if any(kw in name_lower for kw in contract_keywords):
        return DocumentType.CONTRACT
    return DocumentType.RFP


def _build_document_id(filename: str) -> str:
    stem = Path(filename).stem.lower()
    stem = re.sub(r"[\s\-]+", "_", stem)
    stem = re.sub(r"[^a-z0-9_]", "", stem)
    return "DOC_" + stem
