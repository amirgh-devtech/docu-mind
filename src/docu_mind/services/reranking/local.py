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


# Module-level cache: shared across all LocalReranker instances.
# This avoids reloading the model when a new instance is created
# (which happens per RetrievalService — i.e., per request).
_MODEL_CACHE: dict[str, CrossEncoder] = {}


class LocalReranker(BaseReranker):
    """Cross-encoder reranker (sentence-transformers)."""

    name = "local"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def _get_model(self) -> CrossEncoder:
        """Return the cached CrossEncoder, loading it on first use."""
        model_name = self._settings.reranker_model_local
        if model_name not in _MODEL_CACHE:
            from sentence_transformers import CrossEncoder

            logger.info("reranker.loading_model", model=model_name)
            _MODEL_CACHE[model_name] = CrossEncoder(model_name)
            logger.info("reranker.model_loaded", model=model_name)
        return _MODEL_CACHE[model_name]

    def preload(self) -> None:
        """Eager-load the model into the module-level cache."""
        _ = self._get_model()

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Rerank with a CrossEncoder (in a thread to avoid blocking).

        Note:
            CrossEncoder.predict() returns raw logits, which can be
            negative. We apply sigmoid to map them to [0, 1] so that
            scores are interpretable as probabilities.
        """
        if not chunks:
            return []

        import asyncio

        import numpy as np

        model = self._get_model()
        pairs = [(query, c.text_for_llm) for c in chunks]

        loop = asyncio.get_running_loop()
        logits = await loop.run_in_executor(None, model.predict, pairs)

        # Apply sigmoid to map logits → [0, 1]
        logits_arr = np.asarray(logits, dtype=np.float32)
        probs = 1.0 / (1.0 + np.exp(-logits_arr))

        for chunk, prob in zip(chunks, probs, strict=True):
            chunk.score = float(prob)
            chunk.source = "rerank"

        reranked = sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]

        logger.info(
            "reranker.local_done",
            candidates=len(chunks),
            returned=len(reranked),
            score_min=round(float(probs.min()), 4) if len(probs) else 0.0,
            score_max=round(float(probs.max()), 4) if len(probs) else 0.0,
        )
        return reranked
