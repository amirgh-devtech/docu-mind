"""Document upload / list / delete endpoints."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/documents", tags=["documents"])

# Endpoints implemented in a later step.
