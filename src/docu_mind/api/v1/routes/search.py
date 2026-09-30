"""Retrieval-only search endpoints (no LLM)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from docu_mind.api.deps import DBSession, EmbeddingClientDep, SettingsDep
from docu_mind.schemas.search import SearchRequest, SearchResponse

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

    Wired to the retrieval service in step 8.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Retrieval service not yet implemented (step 8)",
    )
