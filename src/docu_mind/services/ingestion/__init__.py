"""Document ingestion (PDF/Word/Markdown → text → chunks → embeddings)."""

from docu_mind.services.ingestion.pipeline import IngestionPipeline

__all__ = ["IngestionPipeline"]
