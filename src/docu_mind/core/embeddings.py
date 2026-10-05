"""Embedding service — provider-agnostic.

Routes to the right SDK based on `EMBEDDING_PROVIDER`:

- ``gemini`` → uses the official ``google-genai`` SDK so that
  ``output_dimensionality`` (Matryoshka truncation) is honoured.
  LiteLLM does not forward this parameter to Gemini, so we bypass it.

- ``openai`` → uses LiteLLM's ``aembedding`` (which passes ``dimensions``
  through to OpenAI's v3 embedding models correctly).

Both paths return the same ``EmbeddingResult`` shape, so no other module
needs to know which provider is active.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    """Async embedding client with batching, retry, and truncation.

    The provider is chosen by ``Settings.embedding_provider``:

    - ``"gemini"`` → Google GenAI SDK (google-genai)
    - ``"openai"`` → LiteLLM's ``aembedding``
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._provider = self._settings.embedding_provider

        # Lazy-loaded provider clients
        self._gemini_client: object | None = None

    @property
    def model(self) -> str:
        return self._settings.active_embedding_model

    @property
    def dimensions(self) -> int:
        return self._settings.active_embedding_dim

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────
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

    async def aclose(self) -> None:
        """Release provider clients."""
        self._gemini_client = None

    # ─────────────────────────────────────────────
    # Dispatch
    # ─────────────────────────────────────────────
    async def _embed_batch(self, batch: list[str]) -> EmbeddingResult:
        """Route to the right provider implementation."""
        if self._provider == "gemini":
            return await self._embed_gemini(batch)
        # provider is guaranteed to be "openai" at this point
        # (enforced by Settings's Literal type)
        return await self._embed_openai(batch)

    # ─────────────────────────────────────────────
    # Gemini (google-genai SDK)
    # ─────────────────────────────────────────────
    def _get_gemini_client(self) -> object:
        """Lazy-load the Google GenAI client."""
        if self._gemini_client is None:
            if not self._settings.gemini_api_key:
                msg = "GEMINI_API_KEY is required for Gemini embeddings"
                raise EmbeddingError(msg)

            from google import genai

            self._gemini_client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._gemini_client

    async def _embed_gemini(self, batch: list[str]) -> EmbeddingResult:
        """Embed via Google GenAI SDK (honours output_dimensionality)."""
        client = self._get_gemini_client()
        model_name = self.model.removeprefix("gemini/")

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._settings.llm_max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((TimeoutError, ConnectionError)),
            reraise=True,
        ):
            with attempt:
                try:
                    response = await client.aio.models.embed_content(  # type: ignore[attr-defined]
                        model=model_name,
                        contents=batch,
                        config={
                            "output_dimensionality": self._settings.active_embedding_dim,
                        },
                    )
                except Exception as e:
                    logger.warning("embedding.gemini_attempt_failed", error=str(e))
                    raise EmbeddingError(f"Gemini embedding failed: {e}") from e

        embeddings = getattr(response, "embeddings", None) or []
        vectors = [list(e.values) for e in embeddings]
        vectors = [_normalise(v) for v in vectors]

        usage = getattr(response, "usage_metadata", None)
        total_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0

        logger.info(
            "embedding.gemini_done",
            count=len(vectors),
            dim=len(vectors[0]) if vectors else 0,
        )

        return EmbeddingResult(
            vectors=vectors,
            model=self.model,
            dimensions=len(vectors[0]) if vectors else self.dimensions,
            total_tokens=total_tokens,
        )

    # ─────────────────────────────────────────────
    # OpenAI (via LiteLLM)
    # ─────────────────────────────────────────────
    async def _embed_openai(self, batch: list[str]) -> EmbeddingResult:
        """Embed via LiteLLM's aembedding (OpenAI)."""
        import litellm

        model = self.model

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
                        dimensions=self._settings.active_embedding_dim,
                    )
                except Exception as e:
                    logger.warning("embedding.openai_attempt_failed", error=str(e))
                    raise EmbeddingError(f"OpenAI embedding failed: {e}") from e

        vectors = [item["embedding"] for item in response.data]
        vectors = [_normalise(v) for v in vectors]

        usage = getattr(response, "usage", None)
        total_tokens = getattr(usage, "total_tokens", 0) if usage else 0

        logger.info(
            "embedding.openai_done",
            count=len(vectors),
            dim=len(vectors[0]) if vectors else 0,
        )

        return EmbeddingResult(
            vectors=vectors,
            model=model,
            dimensions=len(vectors[0]) if vectors else self.dimensions,
            total_tokens=total_tokens,
        )


def _normalise(vec: list[float]) -> list[float]:
    """L2-normalise a vector. Zero vectors are returned unchanged."""
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return [float(x) for x in arr.tolist()]
    return [float(x) for x in (arr / norm).tolist()]
