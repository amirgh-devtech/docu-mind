"""Dense retriever — pgvector HNSW cosine similarity."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from docu_mind.core.embeddings import EmbeddingClient
from docu_mind.core.logging import get_logger
from docu_mind.db.models.chunk import Chunk
from docu_mind.db.models.document import Document
from docu_mind.services.retrieval.base import BaseRetriever, RetrievedChunk

logger = get_logger(__name__)


class DenseRetriever(BaseRetriever):
    """Vector search using pgvector's HNSW index.

    Uses ``<=>`` (cosine distance). We convert distance to similarity
    via ``1 - distance`` so scores are higher-is-better.
    """

    name = "dense"

    def __init__(self, embedder: EmbeddingClient) -> None:
        self._embedder = embedder

    async def retrieve(
        self,
        db: AsyncSession,
        query: str,
        *,
        top_k: int = 20,
    ) -> list[RetrievedChunk]:
        """Embed the query and find nearest neighbors in pgvector."""
        query_vec = await self._embedder.embed_query(query)

        # pgvector operator: <=> = cosine distance (0 = identical, 2 = opposite)
        distance = Chunk.embedding.cosine_distance(query_vec).label("distance")

        stmt = (
            select(Chunk, Document.filename, distance)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.embedding.is_not(None))
            .order_by(distance)
            .limit(top_k)
        )

        result = await db.execute(stmt)
        rows = result.all()

        chunks: list[RetrievedChunk] = []
        for chunk, filename, dist in rows:
            # cosine similarity = 1 - cosine distance
            similarity = 1.0 - float(dist)
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    parent_id=chunk.parent_id,
                    content=chunk.content,
                    context_prefix=chunk.context_prefix,
                    page_number=chunk.page_number,
                    heading_path=chunk.heading_path,
                    token_count=chunk.token_count,
                    score=similarity,
                    document_filename=filename,
                    source=self.name,
                )
            )

        logger.info("retrieval.dense", query_len=len(query), results=len(chunks))
        return chunks
