"""Shared schema primitives."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ORMModel(BaseModel):
    """Base for schemas read from SQLAlchemy models."""

    model_config = ConfigDict(from_attributes=True)


class Pagination(BaseModel):
    """Query params for paginated endpoints."""

    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class Page(BaseModel, Generic[T]):
    """Paginated response wrapper."""

    items: list[T]
    total: int
    limit: int
    offset: int


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None
    code: str | None = None


class MessageResponse(BaseModel):
    """Simple message response."""

    message: str
