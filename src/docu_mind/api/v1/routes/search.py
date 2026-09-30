"""Retrieval-only search endpoints (no LLM)."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/search", tags=["search"])

# Endpoints implemented in a later step.
