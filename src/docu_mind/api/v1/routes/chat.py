"""RAG chat endpoints (with streaming)."""

from __future__ import annotations

from collections.abc import AsyncIterator
import json
from typing import Any

from fastapi import APIRouter, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from docu_mind.api.deps import (
    DBSession,
    EmbeddingClientDep,
    LLMClientDep,
    SettingsDep,
)
from docu_mind.core.exceptions import RAGError
from docu_mind.schemas.chat import ChatRequest, ChatResponse
from docu_mind.services.rag.engine import RAGEngine

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "",
    summary="Ask a question (RAG, non-streaming)",
    response_model=ChatResponse,
)
async def chat(
    payload: ChatRequest,
    settings: SettingsDep,
    db: DBSession,
    llm: LLMClientDep,
    embedder: EmbeddingClientDep,
) -> ChatResponse:
    """Non-streaming RAG chat."""
    engine = RAGEngine(embedder=embedder, settings=settings, llm=llm)

    try:
        result = await engine.answer(
            db,
            payload.query,
            conversation_id=payload.conversation_id,
            top_k=payload.top_k,
            rerank_top_k=payload.rerank_top_k,
            temperature=payload.temperature,
        )
    except RAGError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e

    return ChatResponse(
        conversation_id=result.conversation_id,
        message_id=result.message_id,
        answer=result.answer,
        citations=result.citations,
        retrieved=[],  # not exposed in non-streaming to keep payload small
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=result.latency_ms,
        model=result.model,
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
) -> EventSourceResponse:
    """Streaming RAG chat via Server-Sent Events.

    Events (JSON-encoded in `data:` field):
    - `{"type": "citations", "citations": [...]}`  — sent first
    - `{"type": "token", "content": "..."}`        — many times
    - `{"type": "done", "conversation_id": ..., "message_id": ...}`
    - `{"type": "error", "error": "..."}`
    """
    engine = RAGEngine(embedder=embedder, settings=settings, llm=llm)

    async def event_generator() -> AsyncIterator[dict[str, Any]]:
        async for event in engine.stream(
            db,
            payload.query,
            conversation_id=payload.conversation_id,
            top_k=payload.top_k,
            rerank_top_k=payload.rerank_top_k,
            temperature=payload.temperature,
        ):
            yield {
                "event": event.get("type", "message"),
                "data": json.dumps(event, default=str),
            }

    return EventSourceResponse(event_generator())
