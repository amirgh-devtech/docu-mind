"""Retrieval orchestrator: hybrid → rerank → parent expansion."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from docu_mind.core.config import Settings, get_settings
from docu_mind.core.embeddings import EmbeddingClient
from docu_mind.core.logging import get_logger
from docu_mind.db.models.chunk import Chunk
from docu_mind.services.reranking.service import RerankingService
from docu_mind.services.retrieval.base import RetrievedChunk
from docu_mind.services.retrieval.dense import DenseRetriever
from docu_mind.services.retrieval.hybrid import HybridRetriever
from docu_mind.services.retrieval.keyword import KeywordRetriever

logger = get_logger(__name__)


class RetrievalService:
    """Orchestrates: hybrid retrieval → rerank → parent expansion."""

    def __init__(
        self,
        embedder: EmbeddingClient,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedder = embedder

        # Build retrievers
        self._dense = DenseRetriever(embedder)
        self._keyword = KeywordRetriever(self._settings)
        self._hybrid = HybridRetriever(
            self._dense,
            self._keyword,
            rrf_k=self._settings.rrf_k,
            dense_weight=self._settings.rrf_dense_weight,
            keyword_weight=self._settings.rrf_keyword_weight,
        )

        # Build reranker
        self._reranker = RerankingService(self._settings)

    async def search(
        self,
        db: AsyncSession,
        query: str,
        *,
        top_k: int | None = None,
        rerank_top_k: int | None = None,
        do_rerank: bool = True,
    ) -> list[RetrievedChunk]:
        """Full retrieval pipeline: hybrid → rerank → parent expansion.

        Args:
            db: DB session.
            query: User query.
            top_k: Number of candidates from hybrid (default from settings).
            rerank_top_k: Final number after rerank (default from settings).
            do_rerank: Whether to run the reranker.

        Returns:
            List of RetrievedChunk, ready to feed to the LLM.
        """
        top_k = top_k or self._settings.retrieval_top_k
        rerank_top_k = rerank_top_k or self._settings.rerank_top_k

        # 1. Hybrid retrieval
        candidates = await self._hybrid.retrieve(db, query, top_k=top_k)
        logger.info("retrieval.candidates", count=len(candidates))

        if not candidates:
            return []

        # 2. Reranking
        if do_rerank:
            final = await self._reranker.rerank(query, candidates, top_k=rerank_top_k)
        else:
            final = candidates[:rerank_top_k]

        # 3. Parent expansion (if enabled)
        return await self._expand_parents(db, final)

    async def _expand_parents(
        self,
        db: AsyncSession,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Replace child chunks with their parents (for LLM context).

        If a chunk has a parent_id, we fetch the parent's content and
        swap it in. The parent retains the child's score.
        """
        parent_ids = {c.parent_id for c in chunks if c.parent_id is not None}
        if not parent_ids:
            return chunks

        stmt = select(Chunk).where(Chunk.id.in_(parent_ids))
        result = await db.execute(stmt)
        parents = {p.id: p for p in result.scalars().all()}

        expanded: list[RetrievedChunk] = []
        for c in chunks:
            if c.parent_id and c.parent_id in parents:
                p = parents[c.parent_id]
                expanded.append(
                    RetrievedChunk(
                        chunk_id=p.id,
                        document_id=p.document_id,
                        parent_id=None,  # parent has no parent
                        content=p.content,
                        context_prefix=None,
                        page_number=p.page_number,
                        heading_path=p.heading_path,
                        token_count=p.token_count,
                        score=c.score,
                        document_filename=c.document_filename,
                        source=c.source,
                    )
                )
            else:
                expanded.append(c)

        logger.info(
            "retrieval.parent_expanded",
            original=len(chunks),
            expanded=len(expanded),
        )
        return expanded
