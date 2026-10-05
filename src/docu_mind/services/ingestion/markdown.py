"""Markdown parser using markdown-it-py."""

from __future__ import annotations

from pathlib import Path

from markdown_it import MarkdownIt

from docu_mind.core.exceptions import ParseError
from docu_mind.core.logging import get_logger
from docu_mind.services.ingestion.base import (
    BaseParser,
    ParsedDocument,
    ParsedPage,
)

logger = get_logger(__name__)


class MarkdownParser(BaseParser):
    """Parse Markdown files, tracking heading hierarchy."""

    extensions = ("md", "markdown")

    async def parse(self, path: Path) -> ParsedDocument:
        """Parse a Markdown file.

        Args:
            path: Path to the .md file.

        Returns:
            ParsedDocument with one ParsedPage per heading section.

        Raises:
            ParseError: If the file cannot be read.
        """
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            msg = f"Failed to read Markdown {path.name}: {e}"
            raise ParseError(msg) from e

        md = MarkdownIt("commonmark")
        tokens = md.parse(content)

        pages: list[ParsedPage] = []
        heading_stack: list[tuple[int, str]] = []
        current_lines: list[str] = []

        def flush() -> None:
            text = "\n".join(current_lines).strip()
            if text:
                heading_path = " > ".join(h[1] for h in heading_stack) if heading_stack else None
                pages.append(
                    ParsedPage(
                        text=text,
                        heading_path=heading_path,
                    )
                )
            current_lines.clear()

        i = 0
        while i < len(tokens):
            tok = tokens[i]

            if tok.type == "heading_open":
                # Flush previous section
                flush()
                level = int(tok.tag[1])  # "h2" → 2
                # Next token is the heading text
                if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                    title = tokens[i + 1].content.strip()
                    while heading_stack and heading_stack[-1][0] >= level:
                        heading_stack.pop()
                    heading_stack.append((level, title))
                i += 2  # skip inline + heading_close
                continue

            if tok.type == "inline" and tok.content.strip():
                current_lines.append(tok.content)

            i += 1

        flush()

        metadata: dict[str, object] = {
            "section_count": len(pages),
        }

        logger.info(
            "ingestion.markdown.parsed",
            filename=path.name,
            sections=len(pages),
        )

        return ParsedDocument(pages=pages, metadata=metadata)
