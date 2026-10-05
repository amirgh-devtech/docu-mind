"""Base parser interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ParsedPage:
    """A single page/section of a parsed document."""

    text: str
    page_number: int | None = None
    heading_path: str | None = None


@dataclass(slots=True)
class ParsedDocument:
    """Result of parsing a document."""

    pages: list[ParsedPage]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Concatenate all pages with double newlines."""
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def total_chars(self) -> int:
        """Total character count across pages."""
        return sum(len(p.text) for p in self.pages)


class BaseParser(ABC):
    """Abstract base for document parsers."""

    #: File extensions this parser handles (lowercase, no dot).
    extensions: tuple[str, ...] = ()

    @abstractmethod
    async def parse(self, path: Path) -> ParsedDocument:
        """Parse a file into a structured document.

        Args:
            path: Path to the file on disk.

        Returns:
            ParsedDocument with pages and metadata.

        Raises:
            ParseError: If parsing fails.
        """
        ...
