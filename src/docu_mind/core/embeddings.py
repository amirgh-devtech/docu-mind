"""Embedding service wrapper over LiteLLM (Gemini / OpenAI).

Uses LiteLLM's `embedding()` so we're provider-agnostic. In eco mode
we default to Gemini's `gemini-embedding-001` (free tier).

Handles:
- Batching
- Retry with backoff
- Dimensionality truncation (Matryoshka)
- L2-normalisation for cosine similarity
"""

from __future__ import annotations

from dataclasses import dataclass

import litellm
import numpy as np
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from docu_mind.core.config import Settings, get_settings
from docu_mind.core.exceptions import EmbeddingError
from docu_mind.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class EmbeddingResult:
    """Normalised embedding response."""

    vectors: list[list[float]]
    model: str
    dimensions: int
    total_tokens: int = 0


class EmbeddingClient:
    """Async embedding client with batching, retry, and truncation."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def model(self) -> str:
        return self._settings.active_embedding_model

    @property
    def dimensions(self) -> int:
        return self._settings.active_embedding_dim

    async def embed_texts(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
    ) -> EmbeddingResult:
        """Embed a list of texts, with batching."""
        if not texts:
            return EmbeddingResult(
                vectors=[],
                model=self.model,
                dimensions=self.dimensions,
                total_tokens=0,
            )

        bs = batch_size or self._settings.embedding_batch_size
        all_vectors: list[list[float]] = []
        total_tokens = 0

        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            result = await self._embed_batch(batch)
            all_vectors.extend(result.vectors)
            total_tokens += result.total_tokens

        return EmbeddingResult(
            vectors=all_vectors,
            model=self.model,
            dimensions=self.dimensions,
            total_tokens=total_tokens,
        )

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        result = await self._embed_batch([text])
        return result.vectors[0]

    async def _embed_batch(self, batch: list[str]) -> EmbeddingResult:
        """Embed a single batch with retry/backoff."""
        # For Matryoshka models (Gemini Embedding, OpenAI v3), we can
        # request a smaller dimension.
        model = self.model
        extra: dict[str, object] = {}

        # Gemini Embedding uses `output_dimensionality`
        if model.startswith("gemini/") or model.startswith("text-embedding-3"):
            extra["dimensions"] = self._settings.active_embedding_dim

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._settings.llm_max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((TimeoutError, ConnectionError)),
            reraise=True,
        ):
            with attempt:
                try:
                    response = await litellm.aembedding(
                        model=model,
                        input=batch,
                        **extra,
                    )
                except Exception as e:
                    logger.warning("embedding.attempt_failed", error=str(e))
                    raise EmbeddingError(f"Embedding failed: {e}") from e

        vectors = [item["embedding"] for item in response.data]
        vectors = [_normalise(v) for v in vectors]

        return EmbeddingResult(
            vectors=vectors,
            model=model,
            dimensions=len(vectors[0]) if vectors else self.dimensions,
            total_tokens=getattr(response.usage, "total_tokens", 0),
        )

    async def aclose(self) -> None:
        """LiteLLM doesn't need explicit closing."""
        return


def _normalise(vec: list[float]) -> list[float]:
    """L2-normalise a vector."""
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return [float(x) for x in arr.tolist()]
    return [float(x) for x in (arr / norm).tolist()]
