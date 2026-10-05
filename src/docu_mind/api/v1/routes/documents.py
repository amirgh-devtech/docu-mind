"""Document upload / list / get / delete endpoints."""

from __future__ import annotations

import contextlib
from pathlib import Path
import shutil
from typing import Annotated
import uuid

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select

from docu_mind.api.deps import DBSession, SettingsDep
from docu_mind.core.exceptions import IngestionError, UnsupportedFileTypeError
from docu_mind.db.models.document import Document
from docu_mind.schemas.document import (
    DocumentListResponse,
    DocumentRead,
)
from docu_mind.services.ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown"}
MAX_FILENAME_LEN = 512


@router.post(
    "",
    summary="Upload & ingest a document",
    status_code=status.HTTP_201_CREATED,
    response_model=DocumentRead,
)
async def upload_document(
    settings: SettingsDep,
    db: DBSession,
    file: Annotated[UploadFile, File(description="PDF, DOCX, or Markdown")],
) -> DocumentRead:
    """Upload a document and run the ingestion pipeline synchronously."""
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

    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {suffix!r}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    size_bytes = len(content)
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({size_bytes} bytes). Max: {max_bytes}",
        )

    # Save to upload dir with a unique name
    upload_dir = settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    unique_name = f"{uuid.uuid4()}{suffix}"
    file_path = upload_dir / unique_name

    try:
        file_path.write_bytes(content)
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {e}",
        ) from e

    # Run ingestion
    pipeline = IngestionPipeline(settings)
    try:
        doc = await pipeline.ingest(
            db,
            file_path,
            original_filename=file.filename,
        )
    except UnsupportedFileTypeError as e:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(e),
        ) from e
    except IngestionError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e

    return DocumentRead.model_validate(doc)


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
    total_result = await db.execute(select(func.count()).select_from(Document))
    total = int(total_result.scalar_one())

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
    """Delete a document, its chunks, and the raw file on disk."""
    doc = await db.get(Document, document_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    # Try to delete the raw file (best effort)
    raw_path = Path(doc.storage_path)
    if raw_path.exists() and raw_path.is_file():
        with contextlib.suppress(OSError):
            raw_path.unlink()

    await db.delete(doc)
    await db.commit()


# Silence unused-import warnings (shutil used if we ever stream to disk)
_ = shutil
