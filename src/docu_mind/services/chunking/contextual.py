"""Contextual Retrieval — LLM-generated context prefix per child chunk.

Based on Anthropic's "Contextual Retrieval" technique:
- For each child chunk, ask an LLM to summarise how it fits into the
  larger document.
- Prepend that summary to the chunk before embedding.

This dramatically improves retrieval accuracy (35-49% reduction in
retrieval failure rate).

We use the *fast* LLM (Gemini Flash-Lite in Eco mode) to keep costs low.
"""

from __future__ import annotations

from docu_mind.core.config import Settings, get_settings
from docu_mind.core.llm import LLMClient
from docu_mind.core.logging import get_logger
from docu_mind.services.chunking.base import Chunk

logger = get_logger(__name__)

_CONTEXT_PROMPT = """\
<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:

<chunk>
{chunk}
</chunk>

Please give a short succinct context (2-3 sentences, in the same language as the chunk)
to situate this chunk within the overall document, for the purposes of improving search
retrieval of the chunk. Answer ONLY with the succinct context and nothing else.
"""


class ContextualEnricher:
    """Generates context prefixes for chunks using an LLM."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm or LLMClient(self._settings)

    async def enrich(
        self,
        document_text: str,
        parents: list[Chunk],
        *,
        max_doc_chars: int = 6000,
    ) -> None:
        """Populate `context_prefix` on all child chunks in-place.

        Args:
            document_text: Full document text (truncated internally if too long).
            parents: Parent chunks with children to enrich.
            max_doc_chars: Truncate document text to this many chars for the prompt.
        """
        doc_snippet = document_text[:max_doc_chars]

        for parent in parents:
            for child in parent.children:
                child.context_prefix = await self._generate_context(doc_snippet, child.content)

    async def _generate_context(self, document: str, chunk: str) -> str:
        """Generate a context prefix for a single chunk."""
        prompt = _CONTEXT_PROMPT.format(document=document, chunk=chunk)

        try:
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                kind="fast",
                temperature=0.0,
                max_tokens=150,
            )
            prefix = response.content.strip()
            # Strip potential "Here is..." preambles
            return prefix[:500]  # hard cap
        except Exception as e:
            logger.warning("contextual.enrich_failed", error=str(e))
            return ""
