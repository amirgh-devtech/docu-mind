"""RAG engine: prompts, context, citations, orchestration."""

from docu_mind.services.rag.citations import build_citations, extract_cited_indices
from docu_mind.services.rag.context import BuiltContext, ContextBuilder
from docu_mind.services.rag.engine import RAGEngine, RAGResult
from docu_mind.services.rag.prompts import SYSTEM_PROMPT, build_rag_prompt

__all__ = [
    "SYSTEM_PROMPT",
    "BuiltContext",
    "ContextBuilder",
    "RAGEngine",
    "RAGResult",
    "build_citations",
    "build_rag_prompt",
    "extract_cited_indices",
]
