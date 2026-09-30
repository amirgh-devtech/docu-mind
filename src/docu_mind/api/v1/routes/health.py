"""Health & readiness endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, status
from sqlalchemy import text

from docu_mind import __version__
from docu_mind.api.deps import DBSession, SettingsDep

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="Liveness probe",
    status_code=status.HTTP_200_OK,
)
async def health() -> dict[str, str]:
    """Return a simple liveness response (no DB check)."""
    return {"status": "ok", "version": __version__}


@router.get(
    "/ready",
    summary="Readiness probe (checks DB)",
    status_code=status.HTTP_200_OK,
)
async def ready(db: DBSession) -> dict[str, Any]:
    """Check DB connectivity and pgvector availability."""
    # 1. DB ping
    await db.execute(text("SELECT 1"))

    # 2. pgvector extension check
    result = await db.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))
    pgvector_version = result.scalar()

    return {
        "status": "ready",
        "version": __version__,
        "pgvector": pgvector_version,
    }


@router.get(
    "/config",
    summary="Runtime configuration snapshot (non-secret)",
    status_code=status.HTTP_200_OK,
)
async def config_snapshot(settings: SettingsDep) -> dict[str, Any]:
    """Expose non-secret config values for debugging."""
    return {
        "app_env": settings.app_env,
        "mode": settings.docu_mind_mode,
        "llm_fast": settings.active_llm_fast,
        "llm_smart": settings.active_llm_smart,
        "embedding_model": settings.active_embedding_model,
        "embedding_dim": settings.active_embedding_dim,
        "reranker": settings.active_reranker_provider,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "contextual_chunking": settings.contextual_chunking,
        "retrieval_top_k": settings.retrieval_top_k,
        "rerank_top_k": settings.rerank_top_k,
    }
