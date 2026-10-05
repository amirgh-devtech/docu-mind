"""Context window management for RAG.

Builds numbered context blocks from retrieved chunks while respecting
a token budget for the LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass

from docu_mind.core.config import Settings, get_settings
from docu_mind.core.llm import LLMClient
from docu_mind.core.logging import get_logger
from docu_mind.services.retrieval.base import RetrievedChunk

logger = get_logger(__name__)


@dataclass(slots=True)
class ContextBlock:
    """A single numbered context block."""

    index: int  # 1-based citation index
    chunk: RetrievedChunk
    text: str  # Formatted block: "[1] (source: X)\\n...text..."
    tokens: int


@dataclass(slots=True)
class BuiltContext:
    """Result of building the RAG context."""

    blocks: list[ContextBlock]
    total_tokens: int

    @property
    def text(self) -> str:
        """Concatenated context string for the LLM."""
        return "\n\n".join(b.text for b in self.blocks)

    @property
    def chunk_map(self) -> dict[int, RetrievedChunk]:
        """Map citation index -> chunk."""
        return {b.index: b.chunk for b in self.blocks}


class ContextBuilder:
    """Builds numbered, token-budgeted context from retrieved chunks."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm or LLMClient(self._settings)

    def build(
        self,
        chunks: list[RetrievedChunk],
        *,
        token_budget: int | None = None,
    ) -> BuiltContext:
        """Build numbered context blocks from retrieved chunks.

        Args:
            chunks: Retrieved chunks (already ranked).
            token_budget: Max tokens for the context. If exceeded,
                trailing chunks are dropped.

        Returns:
            BuiltContext with numbered blocks and a chunk_map.
        """
        budget = token_budget or self._settings.rag_context_token_budget
        blocks: list[ContextBlock] = []
        total = 0

        for i, chunk in enumerate(chunks, start=1):
            # Build the source label
            source_parts = [f"source: {chunk.document_filename or 'unknown'}"]
            if chunk.page_number is not None:
                source_parts.append(f"page {chunk.page_number}")
            if chunk.heading_path:
                source_parts.append(f"section: {chunk.heading_path}")
            source_label = ", ".join(source_parts)

            # Full text with prefix
            body = chunk.text_for_llm
            header = f"[{i}] ({source_label})"
            text = f"{header}\n{body}"

            tokens = self._llm.count_tokens(text)

            if total + tokens > budget:
                logger.info(
                    "context.budget_exceeded",
                    kept=i - 1,
                    dropped=len(chunks) - (i - 1),
                    used_tokens=total,
                    budget=budget,
                )
                break

            blocks.append(ContextBlock(index=i, chunk=chunk, text=text, tokens=tokens))
            total += tokens

        logger.info(
            "context.built",
            blocks=len(blocks),
            total_tokens=total,
            budget=budget,
        )

        return BuiltContext(blocks=blocks, total_tokens=total)
