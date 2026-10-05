"""Retrieval-only search endpoints (no LLM)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from docu_mind.api.deps import DBSession, EmbeddingClientDep, SettingsDep
from docu_mind.core.exceptions import RetrievalError
from docu_mind.schemas.chunk import ChunkWithScore
from docu_mind.schemas.search import SearchRequest, SearchResponse
from docu_mind.services.retrieval.service import RetrievalService

router = APIRouter(prefix="/search", tags=["search"])


@router.post(
    "",
    summary="Search documents (no LLM answer)",
    response_model=SearchResponse,
)
async def search(
    payload: SearchRequest,
    settings: SettingsDep,
    db: DBSession,
    embedder: EmbeddingClientDep,
) -> SearchResponse:
    """Retrieve relevant chunks for a query.

    Runs the hybrid retriever + optional reranking + parent expansion.
    No LLM answer is generated — this is a pure retrieval endpoint.
    """
    service = RetrievalService(embedder=embedder, settings=settings)

    try:
        chunks = await service.search(
            db,
            payload.query,
            top_k=payload.top_k,
            rerank_top_k=payload.rerank_top_k,
            do_rerank=payload.rerank,
        )
    except RetrievalError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e

    results = [
        ChunkWithScore(
            id=c.chunk_id,
            document_id=c.document_id,
            parent_id=c.parent_id,
            position=0,  # not exposed at this layer
            content=c.content,
            context_prefix=c.context_prefix,
            page_number=c.page_number,
            heading_path=c.heading_path,
            token_count=c.token_count,
            score=c.score,
            document_filename=c.document_filename,
        )
        for c in chunks
    ]

    return SearchResponse(
        query=payload.query,
        mode=payload.mode,
        results=results,
        total=len(results),
    )
