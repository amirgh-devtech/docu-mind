"""Embedding service wrapper over OpenAI.

Uses the OpenAI SDK directly (not litellm) because embeddings are
provider-locked in practice and the SDK is lighter for batch use.

Handles:
- Batching
- Retry with backoff
- Dimensionality truncation (Pro mode: large → 1024)
- Normalisation for cosine similarity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from openai import APIError, APITimeoutError, AsyncOpenAI
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
        if not self._settings.openai_api_key:
            msg = "OPENAI_API_KEY is required for embeddings"
            raise EmbeddingError(msg)

        self._client = AsyncOpenAI(
            api_key=self._settings.openai_api_key,
            timeout=60.0,
            max_retries=0,  # we handle retries ourselves
        )

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
        """Embed a list of texts, with batching.

        Args:
            texts: List of strings to embed.
            batch_size: Override default batch size.

        Returns:
            EmbeddingResult with a list of vectors, in the same order.

        Raises:
            EmbeddingError: On API failure after retries.
        """
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
        """Embed a single query string.

        Args:
            text: Query text.

        Returns:
            A single embedding vector.
        """
        result = await self._embed_batch([text])
        return result.vectors[0]

    async def _embed_batch(self, batch: list[str]) -> EmbeddingResult:
        """Embed a single batch with retry/backoff."""
        # For Pro mode, we truncate large embeddings to the target dim.
        extra: dict[str, object] = {}
        if self._settings.docu_mind_mode == "pro":
            extra["dimensions"] = self._settings.embedding_dim_pro

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._settings.llm_max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((APITimeoutError, APIError)),
            reraise=True,
        ):
            with attempt:
                try:
                    response = await self._client.embeddings.create(
                        model=self.model,
                        input=batch,
                        **extra,  # type: ignore[arg-type]
                    )
                except (APITimeoutError, APIError) as e:
                    logger.warning("embedding.attempt_failed", error=str(e))
                    raise
                except Exception as e:
                    msg = f"Embedding failed: {e}"
                    raise EmbeddingError(msg) from e

        vectors = [item.embedding for item in response.data]

        # Normalise for cosine similarity (pgvector's <=> uses cosine distance,
        # which is equivalent to L2 on normalised vectors).
        vectors = [_normalise(v) for v in vectors]

        return EmbeddingResult(
            vectors=vectors,
            model=self.model,
            dimensions=len(vectors[0]) if vectors else self.dimensions,
            total_tokens=getattr(response.usage, "total_tokens", 0),
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()


def _normalise(vec: list[float]) -> list[float]:
    """L2-normalise a vector. Zero vectors are returned unchanged."""
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.tolist()
    return cast("list[float]", (arr / norm).tolist())
