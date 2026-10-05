"""Base retriever interface + shared data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import uuid

from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk retrieved from the database, with score and metadata."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    parent_id: uuid.UUID | None
    content: str
    context_prefix: str | None
    page_number: int | None
    heading_path: str | None
    token_count: int
    score: float
    document_filename: str | None = None
    # Which retriever produced this result (dense/keyword/hybrid/rerank)
    source: str = "unknown"

    @property
    def text_for_llm(self) -> str:
        """Return the text to feed to the LLM (prefix + content)."""
        if self.context_prefix:
            return f"{self.context_prefix}\n\n{self.content}"
        return self.content


class BaseRetriever(ABC):
    """Abstract base for retrievers."""

    name: str = "base"

    @abstractmethod
    async def retrieve(
        self,
        db: AsyncSession,
        query: str,
        *,
        top_k: int = 20,
    ) -> list[RetrievedChunk]:
        """Retrieve the top_k chunks for a query.

        Args:
            db: Active DB session.
            query: User query string.
            top_k: Maximum number of results.

        Returns:
            List of RetrievedChunk, ordered by descending score.
        """
        ...
