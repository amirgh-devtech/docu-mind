"""Document schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
import uuid

from pydantic import Field

from docu_mind.schemas.common import ORMModel

DocumentStatus = Literal["pending", "processing", "ready", "failed"]
FileType = Literal["pdf", "docx", "md"]


class DocumentRead(ORMModel):
    """Document metadata returned by the API."""

    id: uuid.UUID
    filename: str
    file_type: FileType
    size_bytes: int
    content_hash: str
    chunk_count: int
    status: DocumentStatus
    error_message: str | None = None
    doc_metadata: dict[str, Any] = Field(default_factory=dict)
    ingested_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(ORMModel):
    """Paginated list of documents."""

    items: list[DocumentRead]
    total: int


class DocumentUploadResponse(ORMModel):
    """Response after a successful upload."""

    id: uuid.UUID
    filename: str
    status: DocumentStatus
    message: str
