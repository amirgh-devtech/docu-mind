"""Aggregate v1 router."""

from __future__ import annotations

from fastapi import APIRouter

api_router = APIRouter(prefix="/api/v1")

# Sub-routers will be included in a later step.
