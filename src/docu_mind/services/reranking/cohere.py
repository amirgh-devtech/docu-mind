"""Cohere Rerank client (Pro mode)."""

from __future__ import annotations

from docu_mind.core.config import Settings, get_settings
from docu_mind.core.exceptions import RerankingError
from docu_mind.core.logging import get_logger
from docu_mind.services.reranking.base import BaseReranker
from docu_mind.services.retrieval.base import RetrievedChunk

logger = get_logger(__name__)


class CohereReranker(BaseReranker):
    """Cohere Rerank v3.5 — hosted cross-encoder with strong accuracy."""

    name = "cohere"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: object | None = None

    def _get_client(self) -> object:
        """Lazy-load the Cohere async client."""
        if self._client is None:
            if not self._settings.cohere_api_key:
                msg = "COHERE_API_KEY is required for Cohere rerank"
                raise RerankingError(msg)

            import cohere

            self._client = cohere.AsyncClient(api_key=self._settings.cohere_api_key)
        return self._client

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Rerank via Cohere's hosted API."""
        if not chunks:
            return []

        client = self._get_client()
        documents = [c.text_for_llm for c in chunks]

        try:
            response = await client.rerank(  # type: ignore[attr-defined]
                model=self._settings.reranker_model_pro,
                query=query,
                documents=documents,
                top_n=top_k,
            )
        except Exception as e:
            logger.warning("reranker.cohere_failed", error=str(e))
            raise RerankingError(f"Cohere rerank failed: {e}") from e

        # Cohere returns results with index + relevance_score
        reranked: list[RetrievedChunk] = []
        for result in response.results:
            chunk = chunks[result.index]
            chunk.score = float(result.relevance_score)
            chunk.source = "rerank"
            reranked.append(chunk)

        logger.info("reranker.cohere_done", candidates=len(chunks), returned=len(reranked))
        return reranked
