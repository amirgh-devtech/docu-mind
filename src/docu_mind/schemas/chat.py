"""Chat schemas."""

from __future__ import annotations

from typing import Literal
import uuid

from pydantic import BaseModel, Field

from docu_mind.schemas.chunk import ChunkWithScore

Role = Literal["user", "assistant", "system"]


class Citation(BaseModel):
    """A citation attached to an assistant message."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_filename: str | None = None
    page_number: int | None = None
    snippet: str
    score: float


class ChatMessage(BaseModel):
    """A single message in a chat request."""

    role: Role
    content: str


class ChatRequest(BaseModel):
    """RAG chat request."""

    query: str = Field(min_length=1, max_length=4000)
    conversation_id: uuid.UUID | None = None
    history: list[ChatMessage] = Field(default_factory=list)
    top_k: int = Field(default=20, ge=1, le=100)
    rerank_top_k: int = Field(default=5, ge=1, le=50)
    stream: bool = True
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    """Non-streaming RAG chat response."""

    conversation_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    citations: list[Citation]
    retrieved: list[ChunkWithScore]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    model: str


class StreamEvent(BaseModel):
    """SSE event payload (JSON-encoded in the `data:` field)."""

    type: Literal["token", "citations", "done", "error"]
    content: str | None = None
    citations: list[Citation] | None = None
    conversation_id: uuid.UUID | None = None
    message_id: uuid.UUID | None = None
    error: str | None = None
    # Usage (only in `done` event)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None
