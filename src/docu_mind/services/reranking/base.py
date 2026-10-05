"""Base reranker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from docu_mind.services.retrieval.base import RetrievedChunk


class BaseReranker(ABC):
    """Abstract base for rerankers."""

    name: str = "base"

    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Re-score and re-order chunks for a query.

        Args:
            query: User query.
            chunks: Candidate chunks from retrieval.
            top_k: Final number of results.

        Returns:
            Reranked list of RetrievedChunk (score field updated).
        """
        ...
