"""RAG chat endpoints (with streaming)."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/chat", tags=["chat"])

# Endpoints implemented in a later step.
