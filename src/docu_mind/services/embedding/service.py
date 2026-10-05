"""Embedding service — wraps EmbeddingClient + DB writes."""

from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from docu_mind.core.config import Settings, get_settings
from docu_mind.core.embeddings import EmbeddingClient, EmbeddingResult
from docu_mind.core.logging import get_logger
from docu_mind.db.models.chunk import Chunk as ChunkModel

logger = get_logger(__name__)


class EmbeddingService:
    """High-level embedding operations tied to the DB."""

    def __init__(
        self,
        client: EmbeddingClient,
        settings: Settings | None = None,
    ) -> None:
        self._client = client
        self._settings = settings or get_settings()

    async def embed_document_chunks(
        self,
        db: AsyncSession,
        chunks: list[ChunkModel],
    ) -> EmbeddingResult:
        """Embed a list of Chunk ORM instances and write vectors back.

        Only *child* chunks (with a `parent_id`) get embedded.
        Parent chunks are skipped.
        """
        children = [c for c in chunks if c.parent_id is not None]
        if not children:
            logger.info("embedding.no_children")
            return EmbeddingResult(
                vectors=[], model=self._client.model, dimensions=self._client.dimensions
            )

        texts = [
            f"{c.context_prefix}\n\n{c.content}" if c.context_prefix else c.content
            for c in children
        ]

        logger.info("embedding.batch_start", count=len(texts))
        result = await self._client.embed_texts(texts)
        logger.info(
            "embedding.batch_done",
            count=len(result.vectors),
            total_tokens=result.total_tokens,
        )

        ts_config = self._settings.keyword_ts_config
        for chunk, vec in zip(children, result.vectors, strict=True):
            chunk.embedding = vec
            chunk.content_tsv = func.to_tsvector(ts_config, chunk.content)

        await db.flush()
        return result
