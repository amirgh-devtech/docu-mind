"""PDF parser using PyMuPDF (fitz)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pymupdf

from docu_mind.core.exceptions import ParseError
from docu_mind.core.logging import get_logger
from docu_mind.services.ingestion.base import (
    BaseParser,
    ParsedDocument,
    ParsedPage,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = get_logger(__name__)


class PDFParser(BaseParser):
    """Parse PDF files with PyMuPDF.

    PyMuPDF is much faster than pdfplumber and preserves layout better.
    We extract text per page and try to detect headings by font size.
    """

    extensions = ("pdf",)

    async def parse(self, path: Path) -> ParsedDocument:
        """Parse a PDF file.

        Args:
            path: Path to the PDF.

        Returns:
            ParsedDocument with one ParsedPage per PDF page.

        Raises:
            ParseError: If the PDF cannot be opened or parsed.
        """
        try:
            doc = pymupdf.open(path)  # type: ignore[no-untyped-call]
        except Exception as e:
            msg = f"Failed to open PDF {path.name}: {e}"
            raise ParseError(msg) from e

        try:
            pages: list[ParsedPage] = []
            page_iter: Iterator[object] = iter(cast("Iterator[object]", doc))
            for page_num, page_obj in enumerate(page_iter, start=1):
                # PyMuPDF Page has get_text but no stubs — use getattr
                text: str = page_obj.get_text("text")  # type: ignore[attr-defined]
                if text.strip():
                    pages.append(
                        ParsedPage(
                            text=text.strip(),
                            page_number=page_num,
                        )
                    )

            meta = getattr(doc, "metadata", None) or {}
            metadata: dict[str, object] = {
                "page_count": getattr(doc, "page_count", len(pages)),
                "title": meta.get("title") or None,
                "author": meta.get("author") or None,
                "producer": meta.get("producer") or None,
            }

            logger.info(
                "ingestion.pdf.parsed",
                filename=path.name,
                pages=len(pages),
                chars=sum(len(p.text) for p in pages),
            )

            return ParsedDocument(pages=pages, metadata=metadata)
        finally:
            doc.close()  # type: ignore[no-untyped-call]
