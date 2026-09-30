"""Chunk schemas."""

from __future__ import annotations

import uuid

from docu_mind.schemas.common import ORMModel


class ChunkRead(ORMModel):
    """Chunk returned from search (no embedding exposed)."""

    id: uuid.UUID
    document_id: uuid.UUID
    parent_id: uuid.UUID | None = None
    position: int
    content: str
    context_prefix: str | None = None
    page_number: int | None = None
    heading_path: str | None = None
    token_count: int


class ChunkWithScore(ChunkRead):
    """Chunk with a retrieval score attached."""

    score: float
    document_filename: str | None = None
