"""Reranker orchestrator — picks the right reranker based on mode."""

from __future__ import annotations

from docu_mind.core.config import Settings, get_settings
from docu_mind.core.logging import get_logger
from docu_mind.services.reranking.base import BaseReranker
from docu_mind.services.reranking.cohere import CohereReranker
from docu_mind.services.reranking.local import LocalReranker
from docu_mind.services.retrieval.base import RetrievedChunk

logger = get_logger(__name__)


class RerankingService:
    """High-level reranking service that picks a provider per mode."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

        # Build both rerankers lazily (they only load models on first use)
        provider = self._settings.active_reranker_provider

        if provider == "cohere":
            self._reranker: BaseReranker = CohereReranker(self._settings)
        else:
            self._reranker = LocalReranker(self._settings)

        logger.info("reranking.service_init", provider=provider)

    @property
    def name(self) -> str:
        return self._reranker.name

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Rerank chunks with the configured provider."""
        return await self._reranker.rerank(query, chunks, top_k=top_k)
