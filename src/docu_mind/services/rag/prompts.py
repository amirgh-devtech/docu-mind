"""Prompt templates for RAG."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a helpful document assistant. Answer the user's question based \
ONLY on the provided context. If the context doesn't contain enough \
information to answer, say so clearly — do NOT make things up.

Rules:
1. Cite sources using bracketed numbers like [1], [2] after each claim.
2. Be concise and factual.
3. If multiple sources support a claim, cite all of them: [1][3].
4. Answer in the same language as the question.
5. Do not mention "context" or "documents" — just answer naturally.
"""


CONTEXT_HEADER = """\
Here is the context from the user's documents. Each source is numbered.

{context}
"""


def build_rag_prompt(
    query: str,
    context_blocks: list[str],
) -> str:
    """Build the user prompt for a RAG query.

    Args:
        query: The user's question.
        context_blocks: List of context blocks, each already numbered
            and formatted (e.g., "[1] (source: doc.md, page 3)\\n...text...").

    Returns:
        Formatted user prompt string.
    """
    context = "\n\n".join(context_blocks)
    return f"""{CONTEXT_HEADER.format(context=context)}

---

Question: {query}"""
