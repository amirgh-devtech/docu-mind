"""Word (.docx) parser using python-docx.

Heading detection strategy (in priority order):
    1. Explicit Word styles: "Heading 1", "Title", "Subtitle"
    2. Formatting heuristics: bold + larger font
    3. Context heuristics: short paragraph followed by a blank line

Falls back gracefully — if nothing matches, paragraphs are grouped
under the nearest preceding heading (or None).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docx import Document as DocxDocument

from docu_mind.core.exceptions import ParseError
from docu_mind.core.logging import get_logger
from docu_mind.services.ingestion.base import (
    BaseParser,
    ParsedDocument,
    ParsedPage,
)

logger = get_logger(__name__)

# python-docx has no complete type stubs, so we use Any for Paragraph.
Paragraph = Any

# Explicit Word style names that indicate headings.
_STYLE_HEADING_MAP: dict[str, int] = {
    "title": 1,
    "subtitle": 2,
}

# Minimum font size (points) to consider a bold paragraph a heading.
_MIN_BOLD_HEADING_SIZE = 14.0

# Maximum length (chars) for the "short paragraph" heuristic.
_MAX_HEADING_LEN = 100


class DocxParser(BaseParser):
    """Parse .docx files into section-based chunks."""

    extensions = ("docx",)

    async def parse(self, path: Path) -> ParsedDocument:
        """Parse a .docx file into section-based ParsedPage items."""
        try:
            doc = DocxDocument(str(path))
        except Exception as e:
            msg = f"Failed to open DOCX {path.name}: {e}"
            raise ParseError(msg) from e

        paragraphs = list(doc.paragraphs)

        # Detect which paragraphs are headings (with level)
        heading_flags: list[int | None] = [
            _detect_heading(paragraphs, i) for i in range(len(paragraphs))
        ]

        # Group into sections
        pages: list[ParsedPage] = []
        heading_stack: list[tuple[int, str]] = []
        buffer: list[str] = []

        def flush_section() -> None:
            """Emit the current buffered section."""
            text = "\n\n".join(buffer).strip()
            if text:
                heading_path = " > ".join(h[1] for h in heading_stack) if heading_stack else None
                pages.append(
                    ParsedPage(
                        text=text,
                        heading_path=heading_path,
                    )
                )
            buffer.clear()

        for para, level in zip(paragraphs, heading_flags, strict=True):
            text = para.text.strip()
            if not text:
                continue

            if level is not None:
                # New heading → flush previous section
                flush_section()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, text))
                continue

            buffer.append(text)

        flush_section()

        metadata: dict[str, object] = {
            "section_count": len(pages),
            "title": doc.core_properties.title or None,
            "author": doc.core_properties.author or None,
        }

        logger.info(
            "ingestion.docx.parsed",
            filename=path.name,
            sections=len(pages),
        )

        return ParsedDocument(pages=pages, metadata=metadata)


# ─────────────────────────────────────────────
# Heading detection
# ─────────────────────────────────────────────
def _detect_heading(
    paragraphs: list[Paragraph],
    idx: int,
) -> int | None:
    """Detect whether the paragraph at `idx` is a heading.

    Returns:
        Heading level (1-9) if heading, None otherwise.
    """
    para = paragraphs[idx]
    text = para.text.strip()
    if not text:
        return None

    # 1. Explicit style
    style_name = (para.style.name or "") if para.style else ""
    level = _level_from_style(style_name)
    if level is not None:
        return level

    # 2. Bold + larger font
    if _is_bold_heading(para):
        return _infer_level_from_context(paragraphs, idx)

    # 3. Short paragraph followed by blank
    if _is_short_heading(paragraphs, idx):
        return _infer_level_from_context(paragraphs, idx)

    return None


def _level_from_style(style_name: str) -> int | None:
    """Extract heading level from Word style name."""
    if not style_name:
        return None

    name = style_name.lower().strip()

    if name in _STYLE_HEADING_MAP:
        return _STYLE_HEADING_MAP[name]

    if name.startswith("heading"):
        parts = name.split()
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                return 1
        return 1

    return None


def _is_bold_heading(para: Paragraph) -> bool:
    """Return True if the paragraph is fully bold and large enough."""
    text = para.text.strip()
    if not text or len(text) > _MAX_HEADING_LEN:
        return False

    # Check bold on all runs
    runs = [r for r in para.runs if r.text.strip()]
    if not runs:
        return False
    if not all(r.bold for r in runs):
        return False

    # Check font size (if provided)
    sizes = [r.font.size.pt for r in runs if r.font.size is not None]
    if not sizes:
        return len(text) < 80

    return bool(max(sizes) >= _MIN_BOLD_HEADING_SIZE)


def _is_short_heading(paragraphs: list[Paragraph], idx: int) -> bool:
    """Return True if paragraph at idx looks like a heading.

    Heuristics:
        - Short (≤ 10 words, ≤ 100 chars)
        - No ending punctuation (. ! ? : ;)
        - Preceded by a blank line OR first non-blank in document
    """
    para = paragraphs[idx]
    text = para.text.strip()

    if not text or len(text) > _MAX_HEADING_LEN:
        return False

    if text.endswith((".", "!", "?", ":", ";")):
        return False

    if len(text.split()) > 10:
        return False

    # First non-blank in document?
    if idx == 0:
        return True

    # Find the previous non-blank paragraph
    j = idx - 1
    while j >= 0 and not paragraphs[j].text.strip():
        j -= 1

    # If we skipped at least one blank paragraph → new section
    return j < 0 or (idx - j) > 1


def _infer_level_from_context(paragraphs: list[Paragraph], idx: int) -> int:
    """Infer heading level when no explicit style is available.

    Uses surrounding indentation/word count as a rough heuristic.
    Defaults to level 2 (most common for section headings).
    """
    _ = paragraphs  # reserved for future use (e.g., font-size analysis)

    # We don't have great signals for level without styles.
    # Default: level 2 (sections). Level 1 is reserved for the document title.
    if idx == 0:
        return 1
    return 2
