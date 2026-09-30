"""Search schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from docu_mind.schemas.chunk import ChunkWithScore

SearchMode = Literal["dense", "keyword", "hybrid"]


class SearchRequest(BaseModel):
    """Retrieval-only search request (no LLM answer)."""

    query: str = Field(min_length=1, max_length=2000)
    mode: SearchMode = "hybrid"
    top_k: int = Field(default=20, ge=1, le=100)
    rerank: bool = True
    rerank_top_k: int = Field(default=5, ge=1, le=50)


class SearchResponse(BaseModel):
    """Search results."""

    query: str
    mode: SearchMode
    results: list[ChunkWithScore]
    total: int
