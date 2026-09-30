"""Aggregate v1 router."""

from __future__ import annotations

from fastapi import APIRouter

from docu_mind.api.v1.routes import chat, documents, health, search

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(search.router)
api_router.include_router(chat.router)
