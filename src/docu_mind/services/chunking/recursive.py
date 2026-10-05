"""Recursive character splitter with heading awareness.

This is a pragmatic implementation that:
1. Splits on paragraph boundaries first (double newline).
2. Falls back to sentence boundaries (. ! ?).
3. Falls back to word boundaries (space).
4. Hard-splits if all else fails.

Produces both parent chunks (for context) and child chunks (for search).
"""

from __future__ import annotations

from docu_mind.core.config import Settings, get_settings
from docu_mind.core.llm import LLMClient
from docu_mind.services.chunking.base import BaseChunker, Chunk

# Separator hierarchy, from most to least preferred
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

# Child chunk size as a fraction of parent chunk size
_CHILD_SIZE_FRACTION = 0.5


class RecursiveChunker(BaseChunker):
    """Recursive character splitter producing parent + child chunks."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm or LLMClient(self._settings)
        self._parent_size = self._settings.chunk_size
        self._overlap = self._settings.chunk_overlap
        self._child_size = max(100, int(self._parent_size * _CHILD_SIZE_FRACTION))

    def chunk(
        self,
        text: str,
        *,
        page_number: int | None = None,
        heading_path: str | None = None,
    ) -> list[Chunk]:
        """Split text into parent + child chunks.

        Args:
            text: Input text.
            page_number: Optional page number.
            heading_path: Optional heading path.

        Returns:
            List of parent chunks, each with `children` populated.
        """
        if not text.strip():
            return []

        # 1. Split into parent chunks
        parent_texts = self._split(text, self._parent_size, self._overlap)

        parents: list[Chunk] = []
        for i, ptext in enumerate(parent_texts):
            parent = Chunk(
                content=ptext,
                position=i,
                token_count=self._llm.count_tokens(ptext),
                page_number=page_number,
                heading_path=heading_path,
            )

            # 2. Split each parent into children
            child_texts = self._split(ptext, self._child_size, self._overlap // 2)
            for j, ctext in enumerate(child_texts):
                parent.children.append(
                    Chunk(
                        content=ctext,
                        position=j,
                        token_count=self._llm.count_tokens(ctext),
                        page_number=page_number,
                        heading_path=heading_path,
                    )
                )

            parents.append(parent)

        return parents

    def _split(self, text: str, size: int, overlap: int) -> list[str]:
        """Recursively split text to chunks of roughly `size` chars.

        Args:
            text: Text to split.
            size: Target chunk size in *characters* (approximation).
            overlap: Overlap between consecutive chunks in characters.

        Returns:
            List of chunk strings.
        """
        # Try each separator from most to least preferred
        for sep in _SEPARATORS:
            if sep == "":
                # Last resort: hard split
                return self._hard_split(text, size, overlap)

            parts = text.split(sep)
            if len(parts) <= 1:
                continue

            # Re-join parts with separator so we don't lose it
            chunks: list[str] = []
            current: list[str] = []
            current_len = 0

            for part in parts:
                part_len = len(part) + len(sep)

                if current_len + part_len > size and current:
                    # Flush current chunk
                    chunks.append(sep.join(current).strip())
                    # Keep overlap: drop from the front until under overlap size
                    while current and current_len > overlap:
                        removed = current.pop(0)
                        current_len -= len(removed) + len(sep)
                    # Continue with remaining + new part
                    current.append(part)
                    current_len += part_len
                else:
                    current.append(part)
                    current_len += part_len

            if current:
                chunks.append(sep.join(current).strip())

            # If any chunk is still too large, recurse with next separator
            if any(len(c) > size * 2 for c in chunks):
                refined: list[str] = []
                for c in chunks:
                    if len(c) > size * 2:
                        refined.extend(self._split(c, size, overlap))
                    else:
                        refined.append(c)
                return [c for c in refined if c.strip()]

            return [c for c in chunks if c.strip()]

        return self._hard_split(text, size, overlap)

    @staticmethod
    def _hard_split(text: str, size: int, overlap: int) -> list[str]:
        """Last-resort hard split on character boundaries."""
        if len(text) <= size:
            return [text.strip()]
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end].strip())
            if end == len(text):
                break
            start = end - overlap
            if start <= 0:
                start = end
        return [c for c in chunks if c]
