"""Base chunker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(slots=True)
class Chunk:
    """A chunk of text before/after splitting.

    Parent chunks have `children` populated after the recursive split;
    children have `parent_index` pointing to their parent.
    """

    content: str
    position: int
    token_count: int
    page_number: int | None = None
    heading_path: str | None = None
    context_prefix: str | None = None
    children: list[Chunk] = field(default_factory=list)


class BaseChunker(ABC):
    """Abstract base for chunkers."""

    @abstractmethod
    def chunk(
        self,
        text: str,
        *,
        page_number: int | None = None,
        heading_path: str | None = None,
    ) -> list[Chunk]:
        """Split text into chunks.

        Args:
            text: Input text.
            page_number: Optional page number for metadata.
            heading_path: Optional heading path for metadata.

        Returns:
            List of chunks.
        """
        ...
