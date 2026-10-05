"""Keyword retriever — PostgreSQL tsvector with ts_rank_cd."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from docu_mind.core.logging import get_logger
from docu_mind.db.models.chunk import Chunk
from docu_mind.db.models.document import Document
from docu_mind.services.retrieval.base import BaseRetriever, RetrievedChunk

logger = get_logger(__name__)


class KeywordRetriever(BaseRetriever):
    """Full-text search using PostgreSQL's tsvector + ts_rank_cd.

    Uses ``plainto_tsquery`` for the query, and ``simple`` config
    (no stemming) which works better for multi-lingual documents.
    """

    name = "keyword"

    async def retrieve(
        self,
        db: AsyncSession,
        query: str,
        *,
        top_k: int = 20,
    ) -> list[RetrievedChunk]:
        """Run keyword search on ``content_tsv``."""
        ts_query = func.plainto_tsquery("simple", query)
        rank = func.ts_rank_cd(Chunk.content_tsv, ts_query).label("rank")

        stmt = (
            select(Chunk, Document.filename, rank)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.content_tsv.op("@@")(ts_query))
            .order_by(rank.desc())
            .limit(top_k)
        )

        result = await db.execute(stmt)
        rows = result.all()

        chunks: list[RetrievedChunk] = []
        for chunk, filename, rank_value in rows:
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
                    score=float(rank_value),
                    document_filename=filename,
                    source=self.name,
                )
            )

        logger.info("retrieval.keyword", query_len=len(query), results=len(chunks))
        return chunks
