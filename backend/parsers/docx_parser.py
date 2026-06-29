"""
DOCX document parser using python-docx.

Implements :class:`core.interfaces.IParser` for `.docx` files.

Content is extracted in *document order* — paragraphs and tables appear
in the same sequence as in the source file.  Tables are converted to a
Markdown representation so that their content is not lost during ingestion.

Pages are emulated by accumulating content until the running character total
reaches or exceeds ``PAGE_CHAR_LIMIT`` (3 000 characters), matching the
project chunk size.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import BinaryIO, List

from docx import Document as _DocxDoc
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table as _DocxTable
from docx.text.paragraph import Paragraph as _DocxParagraph

from core.exceptions import ParseError
from core.interfaces import IParser
from core.models import DocumentType, ParsedDocument
from .pdf_parser import _build_document_id, _infer_document_type

logger = logging.getLogger(__name__)

# Synthetic page size in characters (mirrors settings.max_chunk_chars)
PAGE_CHAR_LIMIT: int = 3_000


class DOCXParser(IParser):
    """Parse DOCX files into :class:`~core.models.ParsedDocument` objects.

    Content is extracted in document order: paragraphs and tables are
    interleaved as they appear in the source, ensuring that table data
    is not silently dropped.  Tables are rendered as Markdown for
    downstream LLM extraction.
    """

    # ------------------------------------------------------------------
    # IParser contract
    # ------------------------------------------------------------------

    def supports(self, filename: str) -> bool:
        """Return ``True`` for any filename ending in ``.docx`` (case-insensitive)."""
        return Path(filename).suffix.lower() == ".docx"

    def parse(self, file: BinaryIO, filename: str, progress_cb=None) -> ParsedDocument:
        """
        Extract paragraph text and table content from a DOCX file and
        group it into synthetic pages of up to ``PAGE_CHAR_LIMIT`` characters
        each.

        Content is extracted in the order it appears in the source document:
        a paragraph followed immediately by a table is preserved in that order.

        Parameters
        ----------
        file:
            Open binary stream of the DOCX, positioned at byte 0.
        filename:
            Original filename; used to derive the document type and ID.

        Returns
        -------
        ParsedDocument
            ``pages`` is a list of page-text strings, each at most
            ``PAGE_CHAR_LIMIT`` characters (unless a single block
            exceeds that limit by itself).

        Raises
        ------
        core.exceptions.ParseError
            Wraps any underlying :mod:`docx` or I/O error.
        """
        try:
            raw: bytes = file.read()
            docx_doc = _DocxDoc(io.BytesIO(raw))

            # Walk the document body in order, collecting text blocks
            blocks: List[str] = _extract_blocks_in_order(docx_doc)
            pages = _group_into_pages(blocks, PAGE_CHAR_LIMIT)

            doc_type: DocumentType = _infer_document_type(filename)
            doc_id: str = _build_document_id(filename)

            logger.info(
                "Parsed DOCX '%s': %d content blocks → %d synthetic pages, type=%s",
                filename, len(blocks), len(pages), doc_type.value,
            )

            return ParsedDocument(
                id=doc_id,
                name=filename,
                type=doc_type,
                pages=pages,
                total_pages=len(pages),
            )

        except Exception as exc:
            raise ParseError(
                f"Failed to parse DOCX '{filename}': {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_blocks_in_order(docx_doc: _DocxDoc) -> List[str]:
    """Walk the document body XML and return content blocks in document order.

    For each child of ``<w:body>`` that is either a paragraph (``<w:p>``) or a
    table (``<w:tbl>``), the corresponding text is extracted and appended.
    This preserves the interleaving of paragraphs and tables as they appear
    in the source, unlike iterating ``docx_doc.paragraphs`` which skips tables.

    Parameters
    ----------
    docx_doc:
        An opened :class:`docx.Document` object.

    Returns
    -------
    list[str]
        Non-empty text blocks, each representing either a paragraph or a
        Markdown-formatted table.
    """
    blocks: List[str] = []

    for child in docx_doc.element.body:
        if isinstance(child, CT_P):
            # Paragraph element
            para = _DocxParagraph(child, docx_doc)
            text = para.text.strip()
            if text:
                blocks.append(text)

        elif isinstance(child, CT_Tbl):
            # Table element
            table = _DocxTable(child, docx_doc)
            md = _table_to_markdown(table)
            if md.strip():
                blocks.append(md)

    return blocks


def _table_to_markdown(table: _DocxTable) -> str:
    """Convert a DOCX table to a Markdown-formatted string.

    Merged cells are represented by their visible text (python-docx exposes
    them as duplicated cells, so the raw cell text is de-duplicated within
    each row to avoid repetition).

    Parameters
    ----------
    table:
        A :class:`docx.table.Table` object.

    Returns
    -------
    str
        Markdown table string, including a header separator after the first
        row, or an empty string if the table has no usable rows.
    """
    rows: List[str] = []
    for row_idx, row in enumerate(table.rows):
        # Deduplicate consecutive identical cells (result of column merges)
        seen_texts: List[str] = []
        for cell in row.cells:
            text = cell.text.strip().replace("\n", " ")
            if not seen_texts or seen_texts[-1] != text:
                seen_texts.append(text)
        row_str = "| " + " | ".join(seen_texts) + " |"
        rows.append(row_str)
        if row_idx == 0:
            rows.append("| " + " | ".join(["---"] * len(seen_texts)) + " |")

    if not rows:
        return ""
    return "\n[TABLE]\n" + "\n".join(rows) + "\n"


def _group_into_pages(blocks: List[str], limit: int) -> List[str]:
    """Accumulate *blocks* into synthetic pages of at most *limit* characters.

    A new page is started when the current page's character count reaches
    or exceeds *limit*.  Each block is separated from the previous one
    by a newline.  An empty block list produces a single empty-string page
    so that ``ParsedDocument`` invariants still hold.

    Parameters
    ----------
    blocks:
        Non-empty stripped text blocks (paragraphs or Markdown tables).
    limit:
        Character threshold at which to close the current page and open a
        new one.

    Returns
    -------
    list[str]
        At least one element.
    """
    if not blocks:
        return [""]

    pages: List[str] = []
    current_chunks: List[str] = []
    current_len: int = 0

    for block in blocks:
        if current_len >= limit and current_chunks:
            pages.append("\n".join(current_chunks))
            current_chunks = []
            current_len = 0

        current_chunks.append(block)
        current_len += len(block)

    # Flush the final page
    if current_chunks:
        pages.append("\n".join(current_chunks))

    return pages
