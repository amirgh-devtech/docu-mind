"""Domain-specific exceptions."""

from __future__ import annotations


class DocuMindError(Exception):
    """Base class for all docu-mind errors."""


# ─────────────────────────────────────────────
# Config / setup
# ─────────────────────────────────────────────
class ConfigurationError(DocuMindError):
    """Raised when configuration is invalid or incomplete."""


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
class LLMError(DocuMindError):
    """Base for all LLM-related errors."""


class LLMProviderError(LLMError):
    """Raised when an LLM provider call fails after retries."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM call times out."""


# ─────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────
class EmbeddingError(DocuMindError):
    """Raised when embedding fails."""


# ─────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────
class IngestionError(DocuMindError):
    """Raised when document ingestion fails."""


class UnsupportedFileTypeError(IngestionError):
    """Raised for unsupported document types."""


class ParseError(IngestionError):
    """Raised when a parser fails on a document."""


# ─────────────────────────────────────────────
# Retrieval / RAG
# ─────────────────────────────────────────────
class RetrievalError(DocuMindError):
    """Raised when retrieval fails."""


class RerankingError(DocuMindError):
    """Raised when reranking fails."""


class RAGError(DocuMindError):
    """Raised when the RAG engine fails."""


# ─────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────
class StorageError(DocuMindError):
    """Raised for storage / DB errors."""


class DocumentNotFoundError(StorageError):
    """Raised when a document ID does not exist."""
