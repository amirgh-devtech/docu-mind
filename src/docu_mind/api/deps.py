"""FastAPI dependency providers.

Centralises access to settings, DB session, and the various service
clients (LLM, embeddings, reranker). All clients are lazily constructed
and cached on the app state so we don't rebuild them per-request.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from docu_mind.core.config import Settings, get_settings
from docu_mind.core.embeddings import EmbeddingClient
from docu_mind.core.llm import LLMClient
from docu_mind.db.session import get_session


# ─────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────
def get_settings_dep() -> Settings:
    """Return the application settings (cached)."""
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


# ─────────────────────────────────────────────
# DB session
# ─────────────────────────────────────────────
async def get_db() -> AsyncIterator[AsyncSession]:
    """Yield an async DB session."""
    async for session in get_session():
        yield session


DBSession = Annotated[AsyncSession, Depends(get_db)]


# ─────────────────────────────────────────────
# Clients (lazy singletons on app.state)
# ─────────────────────────────────────────────
def get_llm_client(request: Request) -> LLMClient:
    """Return the shared LLM client from app state."""
    client: LLMClient = request.app.state.llm_client
    return client


def get_embedding_client(request: Request) -> EmbeddingClient:
    """Return the shared embedding client from app state."""
    client: EmbeddingClient = request.app.state.embedding_client
    return client


LLMClientDep = Annotated[LLMClient, Depends(get_llm_client)]
EmbeddingClientDep = Annotated[EmbeddingClient, Depends(get_embedding_client)]
