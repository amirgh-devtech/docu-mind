"""Citation extraction from LLM responses.

Parses [N] markers in the LLM's answer and maps them back to the
context chunks that were provided.
"""

from __future__ import annotations

import re

from docu_mind.core.logging import get_logger
from docu_mind.schemas.chat import Citation
from docu_mind.services.rag.context import BuiltContext
from docu_mind.services.retrieval.base import RetrievedChunk

logger = get_logger(__name__)

# Matches [1], [2], [12], [1][3], [1,3], [1, 3]
_CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_cited_indices(answer: str) -> set[int]:
    """Extract all unique citation indices from an LLM answer.

    Args:
        answer: The LLM's answer text with [N] markers.

    Returns:
        Set of integer indices that were cited.
    """
    indices: set[int] = set()
    for match in _CITATION_PATTERN.finditer(answer):
        # match.group(1) can be "1" or "1, 3" or "1,3,5"
        for part in match.group(1).split(","):
            part = part.strip()
            if part.isdigit():
                indices.add(int(part))
    return indices


def build_citations(
    answer: str,
    context: BuiltContext,
) -> list[Citation]:
    """Build a list of Citation objects from the answer + context.

    Only returns citations for chunks that the LLM actually referenced.

    Args:
        answer: The LLM's answer text.
        context: The BuiltContext used for the LLM call.

    Returns:
        List of Citation objects.
    """
    cited_indices = extract_cited_indices(answer)
    if not cited_indices:
        logger.info("citations.none_found")
        return []

    chunk_map = context.chunk_map
    citations: list[Citation] = []

    for idx in sorted(cited_indices):
        chunk = chunk_map.get(idx)
        if chunk is None:
            # LLM hallucinated a citation number
            logger.warning("citations.invalid_index", index=idx)
            continue
        citations.append(_chunk_to_citation(chunk))

    logger.info(
        "citations.extracted",
        cited=len(citations),
        available=len(chunk_map),
    )
    return citations


def _chunk_to_citation(chunk: RetrievedChunk) -> Citation:
    """Convert a RetrievedChunk to a Citation."""
    # Snippet: first 200 chars of content
    snippet = chunk.content[:200]
    if len(chunk.content) > 200:
        snippet += "..."

    return Citation(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document_filename=chunk.document_filename,
        page_number=chunk.page_number,
        snippet=snippet,
        score=chunk.score,
    )
