"""Word (.docx) parser using python-docx."""

from __future__ import annotations

from pathlib import Path

from docx import Document as DocxDocument

from docu_mind.core.exceptions import ParseError
from docu_mind.core.logging import get_logger
from docu_mind.services.ingestion.base import (
    BaseParser,
    ParsedDocument,
    ParsedPage,
)

logger = get_logger(__name__)


class DocxParser(BaseParser):
    """Parse .docx files into paragraphs.

    Word doesn't have a concept of "pages" that maps to text layout,
    so we treat each paragraph as a unit and infer heading paths
    from the paragraph style (Heading 1, Heading 2, ...).
    """

    extensions = ("docx",)

    async def parse(self, path: Path) -> ParsedDocument:
        """Parse a .docx file.

        Args:
            path: Path to the .docx file.

        Returns:
            ParsedDocument with one ParsedPage per paragraph.

        Raises:
            ParseError: If the file cannot be parsed.
        """
        try:
            doc = DocxDocument(str(path))
        except Exception as e:
            msg = f"Failed to open DOCX {path.name}: {e}"
            raise ParseError(msg) from e

        pages: list[ParsedPage] = []
        heading_stack: list[tuple[int, str]] = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            style_name = (para.style.name or "") if para.style else ""
            heading_level = _heading_level(style_name)

            if heading_level is not None:
                # Maintain a stack of (level, text)
                while heading_stack and heading_stack[-1][0] >= heading_level:
                    heading_stack.pop()
                heading_stack.append((heading_level, text))

            heading_path = " > ".join(h[1] for h in heading_stack) if heading_stack else None

            pages.append(
                ParsedPage(
                    text=text,
                    heading_path=heading_path,
                )
            )

        metadata: dict[str, object] = {
            "paragraph_count": len(pages),
            "title": doc.core_properties.title or None,
            "author": doc.core_properties.author or None,
        }

        logger.info(
            "ingestion.docx.parsed",
            filename=path.name,
            paragraphs=len(pages),
        )

        return ParsedDocument(pages=pages, metadata=metadata)


def _heading_level(style_name: str) -> int | None:
    """Return heading level (1-9) for a Word style name, or None."""
    name = style_name.lower()
    if not name.startswith("heading"):
        return None
    # "Heading 1" → 1, "Heading 2" → 2, ...
    parts = name.split()
    if len(parts) < 2:
        return 1
    try:
        return int(parts[1])
    except ValueError:
        return 1
