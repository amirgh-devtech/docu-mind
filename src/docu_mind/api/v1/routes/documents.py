"""Document upload / list / get / delete endpoints.

The heavy lifting (parsing, chunking, embedding) happens in
`services.ingestion.pipeline` — wired up in step 7. For now these
endpoints validate input and return 501 for the not-yet-implemented
pipeline.
"""

from __future__ import annotations

from typing import Annotated
import uuid

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select

from docu_mind.api.deps import DBSession, SettingsDep
from docu_mind.db.models.document import Document
from docu_mind.schemas.document import (
    DocumentListResponse,
    DocumentRead,
)

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown"}
MAX_FILENAME_LEN = 512


@router.post(
    "",
    summary="Upload a document",
    status_code=status.HTTP_201_CREATED,
    response_model=DocumentRead,
)
async def upload_document(
    settings: SettingsDep,
    db: DBSession,
    file: Annotated[UploadFile, File(description="PDF, DOCX, or Markdown")],
) -> DocumentRead:
    """Upload a document for ingestion.

    Currently validates and stores metadata; the actual ingestion
    pipeline lands in step 7.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    if len(file.filename) > MAX_FILENAME_LEN:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Filename too long (max {MAX_FILENAME_LEN})",
        )

    # Extension check
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {suffix!r}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Size check (reads the file into memory — fine for our max size)
    content = await file.read()
    size_bytes = len(content)
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({size_bytes} bytes). Max: {max_bytes}",
        )

    # Placeholder: pipeline will be called in step 7.
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Ingestion pipeline not yet implemented (step 7)",
    )


@router.get(
    "",
    summary="List documents",
    response_model=DocumentListResponse,
)
async def list_documents(
    db: DBSession,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentListResponse:
    """List documents, newest first."""
    # Total count
    total_result = await db.execute(select(func.count()).select_from(Document))
    total = int(total_result.scalar_one())

    # Page
    stmt = select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
    result = await db.execute(stmt)
    docs = list(result.scalars().all())

    return DocumentListResponse(
        items=[DocumentRead.model_validate(d) for d in docs],
        total=total,
    )


@router.get(
    "/{document_id}",
    summary="Get a document by ID",
    response_model=DocumentRead,
)
async def get_document(document_id: uuid.UUID, db: DBSession) -> DocumentRead:
    """Fetch a single document by ID."""
    doc = await db.get(Document, document_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )
    return DocumentRead.model_validate(doc)


@router.delete(
    "/{document_id}",
    summary="Delete a document",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document(document_id: uuid.UUID, db: DBSession) -> None:
    """Delete a document and all its chunks (cascade)."""
    doc = await db.get(Document, document_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )
    await db.delete(doc)
    await db.commit()
