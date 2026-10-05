"""Local cross-encoder reranker using sentence-transformers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from docu_mind.core.config import Settings, get_settings
from docu_mind.core.logging import get_logger
from docu_mind.services.reranking.base import BaseReranker
from docu_mind.services.retrieval.base import RetrievedChunk

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = get_logger(__name__)


class LocalReranker(BaseReranker):
    """Cross-encoder reranker (sentence-transformers).

    Uses ``cross-encoder/ms-marco-MiniLM-L6-v2`` by default — small,
    fast, and works on CPU. Loaded lazily on first use.
    """

    name = "local"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        """Lazy-load the CrossEncoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(
                "reranker.loading_model",
                model=self._settings.reranker_model_local,
            )
            self._model = CrossEncoder(self._settings.reranker_model_local)
        return self._model

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Rerank with a CrossEncoder (in a thread to avoid blocking)."""
        if not chunks:
            return []

        import asyncio

        model = self._get_model()

        # Build (query, passage) pairs. Use text_for_llm so we include context.
        pairs = [(query, c.text_for_llm) for c in chunks]

        # CrossEncoder.predict is sync + CPU-bound → run in a thread
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(None, model.predict, pairs)

        # Assign scores and sort
        for chunk, score in zip(chunks, scores, strict=True):
            chunk.score = float(score)
            chunk.source = "rerank"

        reranked = sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]

        logger.info("reranker.local_done", candidates=len(chunks), returned=len(reranked))
        return reranked
