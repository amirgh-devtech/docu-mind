"""End-to-end ingestion pipeline.

Orchestrates: parse → chunk → contextual enrich → embed → save.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
from pathlib import Path
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from docu_mind.core.config import Settings, get_settings
from docu_mind.core.embeddings import EmbeddingClient
from docu_mind.core.exceptions import IngestionError, UnsupportedFileTypeError
from docu_mind.core.llm import LLMClient
from docu_mind.core.logging import get_logger
from docu_mind.db.models.chunk import Chunk as ChunkModel
from docu_mind.db.models.document import Document
from docu_mind.services.chunking.base import Chunk as ChunkData
from docu_mind.services.chunking.contextual import ContextualEnricher
from docu_mind.services.chunking.recursive import RecursiveChunker
from docu_mind.services.embedding.service import EmbeddingService
from docu_mind.services.ingestion.base import BaseParser
from docu_mind.services.ingestion.docx import DocxParser
from docu_mind.services.ingestion.markdown import MarkdownParser
from docu_mind.services.ingestion.pdf import PDFParser

logger = get_logger(__name__)


def _select_parser(path: Path) -> BaseParser:
    """Return the right parser for a file based on its extension."""
    ext = path.suffix.lower().lstrip(".")
    parsers: list[BaseParser] = [PDFParser(), DocxParser(), MarkdownParser()]
    for parser in parsers:
        if ext in parser.extensions:
            return parser
    msg = f"Unsupported file type: .{ext}"
    raise UnsupportedFileTypeError(msg)


def _file_hash(path: Path) -> str:
    """Compute SHA-256 of file content."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


class IngestionPipeline:
    """Orchestrates document ingestion end-to-end."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        embedder: EmbeddingClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm or LLMClient(self._settings)
        self._embedder = embedder or EmbeddingClient(self._settings)
        self._chunker = RecursiveChunker(self._settings, self._llm)
        self._enricher = ContextualEnricher(self._settings, self._llm)
        self._embedding_service = EmbeddingService(self._embedder)

    async def ingest(
        self,
        db: AsyncSession,
        file_path: Path,
        *,
        original_filename: str | None = None,
    ) -> Document:
        """Ingest a file into the database.

        Args:
            db: Active DB session.
            file_path: Path to the file on disk.
            original_filename: Original filename (if different from path.name).

        Returns:
            The created Document.

        Raises:
            IngestionError: If any step fails.
        """
        filename = original_filename or file_path.name
        content_hash = _file_hash(file_path)
        size_bytes = file_path.stat().st_size
        file_type = file_path.suffix.lower().lstrip(".")

        # 1. Dedup check
        existing = await db.execute(select(Document).where(Document.content_hash == content_hash))
        existing_doc = existing.scalar_one_or_none()
        if existing_doc is not None:
            logger.info(
                "ingestion.duplicate",
                filename=filename,
                existing_id=str(existing_doc.id),
            )
            return existing_doc

        # 2. Create Document row (pending)
        doc = Document(
            id=uuid.uuid4(),
            filename=filename,
            file_type=file_type,
            size_bytes=size_bytes,
            content_hash=content_hash,
            storage_path=str(file_path),
            status="processing",
            doc_metadata={},
        )
        db.add(doc)
        await db.flush()

        try:
            # 3. Parse
            parser = _select_parser(file_path)
            parsed = await parser.parse(file_path)
            logger.info(
                "ingestion.parsed",
                filename=filename,
                pages=len(parsed.pages),
                chars=parsed.total_chars,
            )

            # 4. Chunk each page → parent + child
            all_parents: list[ChunkData] = []
            for page in parsed.pages:
                parents = self._chunker.chunk(
                    page.text,
                    page_number=page.page_number,
                    heading_path=page.heading_path,
                )
                all_parents.extend(parents)

            logger.info(
                "ingestion.chunked",
                parents=len(all_parents),
                children=sum(len(p.children) for p in all_parents),
            )

            # 5. Contextual enrichment (optional)
            if self._settings.contextual_chunking:
                await self._enricher.enrich(parsed.full_text, all_parents)
                logger.info("ingestion.contextual_done")

            # 6. Persist parents + children as ORM rows
            orm_chunks: list[ChunkModel] = []
            for parent_data in all_parents:
                parent_orm = ChunkModel(
                    id=uuid.uuid4(),
                    document_id=doc.id,
                    parent_id=None,
                    position=parent_data.position,
                    content=parent_data.content,
                    context_prefix=None,
                    token_count=parent_data.token_count,
                    page_number=parent_data.page_number,
                    heading_path=parent_data.heading_path,
                    embedding=None,
                    content_tsv=None,
                )
                db.add(parent_orm)

                for child_data in parent_data.children:
                    child_orm = ChunkModel(
                        id=uuid.uuid4(),
                        document_id=doc.id,
                        parent_id=parent_orm.id,
                        position=child_data.position,
                        content=child_data.content,
                        context_prefix=child_data.context_prefix,
                        token_count=child_data.token_count,
                        page_number=child_data.page_number,
                        heading_path=child_data.heading_path,
                        embedding=None,
                        content_tsv=None,
                    )
                    db.add(child_orm)
                    orm_chunks.append(child_orm)
                orm_chunks.append(parent_orm)

            await db.flush()

            # 7. Embed children + write vectors + tsvector
            await self._embedding_service.embed_document_chunks(db, orm_chunks)

            # 8. Finalise document
            doc.status = "ready"
            doc.chunk_count = len(orm_chunks)
            doc.ingested_at = datetime.now(UTC)
            doc.doc_metadata = parsed.metadata

            await db.commit()
            await db.refresh(doc)

            logger.info(
                "ingestion.complete",
                document_id=str(doc.id),
                chunks=doc.chunk_count,
            )
            return doc

        except Exception as e:
            await db.rollback()
            # Re-fetch in a new transaction to mark as failed
            doc.status = "failed"
            doc.error_message = str(e)[:2000]
            db.add(doc)
            await db.commit()
            logger.exception("ingestion.failed", filename=filename)
            raise IngestionError(f"Ingestion failed for {filename}: {e}") from e
