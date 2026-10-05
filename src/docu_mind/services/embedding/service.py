"""Embedding service — wraps EmbeddingClient + DB writes."""

from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from docu_mind.core.embeddings import EmbeddingClient, EmbeddingResult
from docu_mind.core.logging import get_logger
from docu_mind.db.models.chunk import Chunk as ChunkModel

logger = get_logger(__name__)


class EmbeddingService:
    """High-level embedding operations tied to the DB."""

    def __init__(self, client: EmbeddingClient) -> None:
        self._client = client

    async def embed_document_chunks(
        self,
        db: AsyncSession,
        chunks: list[ChunkModel],
    ) -> EmbeddingResult:
        """Embed a list of Chunk ORM instances and write vectors back.

        Only *child* chunks (with a `parent_id`) get embedded.
        Parent chunks are skipped.

        Args:
            db: Active DB session.
            chunks: Chunk ORM models to embed.

        Returns:
            EmbeddingResult with the embeddings.
        """
        # Filter to children only
        children = [c for c in chunks if c.parent_id is not None]
        if not children:
            logger.info("embedding.no_children")
            return EmbeddingResult(
                vectors=[], model=self._client.model, dimensions=self._client.dimensions
            )

        # Build the text we actually embed: context_prefix + content
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

        # Write vectors + tsvector back
        for chunk, vec in zip(children, result.vectors, strict=True):
            chunk.embedding = vec
            chunk.content_tsv = func.to_tsvector("simple", chunk.content)

        await db.flush()
        return result
