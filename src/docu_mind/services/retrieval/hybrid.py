"""Hybrid retriever — RRF fusion of dense + keyword results."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from docu_mind.core.logging import get_logger
from docu_mind.services.retrieval.base import BaseRetriever, RetrievedChunk
from docu_mind.services.retrieval.dense import DenseRetriever
from docu_mind.services.retrieval.keyword import KeywordRetriever

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """Combines dense + keyword via Reciprocal Rank Fusion (RRF).

    RRF score for a document d = Σ over retrievers r of 1/(k + rank_r(d))

    where k is a smoothing constant (default 60, from the original paper).
    RRF is robust because it doesn't need score normalization between
    retrievers with different score scales.
    """

    name = "hybrid"

    def __init__(
        self,
        dense: DenseRetriever,
        keyword: KeywordRetriever,
        *,
        rrf_k: int = 60,
        dense_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> None:
        self._dense = dense
        self._keyword = keyword
        self._rrf_k = rrf_k
        self._dense_weight = dense_weight
        self._keyword_weight = keyword_weight

    async def retrieve(
        self,
        db: AsyncSession,
        query: str,
        *,
        top_k: int = 20,
    ) -> list[RetrievedChunk]:
        """Run dense + keyword in parallel-ish, then fuse via RRF."""
        # Run both retrievers with a larger candidate pool
        candidate_k = max(top_k * 2, 30)

        dense_results = await self._dense.retrieve(db, query, top_k=candidate_k)
        keyword_results = await self._keyword.retrieve(db, query, top_k=candidate_k)

        # Build rank maps: chunk_id → rank (1-indexed)
        dense_ranks = {c.chunk_id: i + 1 for i, c in enumerate(dense_results)}
        keyword_ranks = {c.chunk_id: i + 1 for i, c in enumerate(keyword_results)}

        # Union of all chunk IDs
        all_ids = set(dense_ranks) | set(keyword_ranks)

        # Build a lookup for full chunk data
        chunk_map: dict[uuid.UUID, RetrievedChunk] = {}
        for c in dense_results:
            chunk_map[c.chunk_id] = c
        for c in keyword_results:
            chunk_map.setdefault(c.chunk_id, c)

        # Compute RRF score
        rrf_scores: dict[uuid.UUID, float] = {}
        for cid in all_ids:
            score = 0.0
            if cid in dense_ranks:
                score += self._dense_weight / (self._rrf_k + dense_ranks[cid])
            if cid in keyword_ranks:
                score += self._keyword_weight / (self._rrf_k + keyword_ranks[cid])
            rrf_scores[cid] = score

        # Sort by descending RRF score
        sorted_ids = sorted(
            rrf_scores,
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )[:top_k]

        fused: list[RetrievedChunk] = []
        for cid in sorted_ids:
            c = chunk_map[cid]
            c.score = rrf_scores[cid]
            c.source = "hybrid"
            fused.append(c)

        logger.info(
            "retrieval.hybrid",
            dense=len(dense_results),
            keyword=len(keyword_results),
            fused=len(fused),
        )
        return fused
