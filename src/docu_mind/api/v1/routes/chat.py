"""RAG chat endpoints (with streaming)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from docu_mind.api.deps import (
    DBSession,
    EmbeddingClientDep,
    LLMClientDep,
    SettingsDep,
)
from docu_mind.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "",
    summary="Ask a question (RAG)",
    response_model=ChatResponse,
)
async def chat(
    payload: ChatRequest,
    settings: SettingsDep,
    db: DBSession,
    llm: LLMClientDep,
    embedder: EmbeddingClientDep,
) -> ChatResponse:
    """Non-streaming RAG chat.

    Set `stream=false` in the request for this path.
    Wired to the RAG engine in step 9.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="RAG engine not yet implemented (step 9)",
    )


@router.post(
    "/stream",
    summary="Ask a question (RAG, SSE streaming)",
)
async def chat_stream(
    payload: ChatRequest,
    settings: SettingsDep,
    db: DBSession,
    llm: LLMClientDep,
    embedder: EmbeddingClientDep,
) -> None:
    """Streaming RAG chat via Server-Sent Events.

    Wired to the RAG engine in step 9.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="RAG streaming not yet implemented (step 9)",
    )
