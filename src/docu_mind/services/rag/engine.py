"""RAG orchestrator: retrieval → context → LLM → citations."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
import time
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from docu_mind.core.config import Settings, get_settings
from docu_mind.core.embeddings import EmbeddingClient
from docu_mind.core.llm import LLMClient
from docu_mind.core.logging import get_logger
from docu_mind.db.models.conversation import Conversation
from docu_mind.db.models.message import Message
from docu_mind.schemas.chat import Citation
from docu_mind.services.rag.citations import build_citations
from docu_mind.services.rag.context import ContextBuilder
from docu_mind.services.rag.prompts import SYSTEM_PROMPT, build_rag_prompt
from docu_mind.services.retrieval.base import RetrievedChunk
from docu_mind.services.retrieval.service import RetrievalService

logger = get_logger(__name__)


@dataclass(slots=True)
class RAGResult:
    """Non-streaming RAG result."""

    answer: str
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


class RAGEngine:
    """End-to-end RAG orchestrator."""

    def __init__(
        self,
        embedder: EmbeddingClient,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm or LLMClient(self._settings)
        self._embedder = embedder
        self._retrieval = RetrievalService(embedder, self._settings)
        self._context = ContextBuilder(self._settings, self._llm)

    # ─────────────────────────────────────────────
    # Non-streaming
    # ─────────────────────────────────────────────
    async def answer(
        self,
        db: AsyncSession,
        query: str,
        *,
        conversation_id: uuid.UUID | None = None,
        top_k: int | None = None,
        rerank_top_k: int | None = None,
        temperature: float | None = None,
    ) -> RAGResult:
        """Run a RAG query end-to-end (non-streaming)."""
        start = time.monotonic()

        # 0. Load conversation history (if continuing)
        history = await self._load_history(db, conversation_id)

        # 1. Retrieve
        chunks = await self._retrieval.search(
            db,
            query,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
        )

        # 2. Build context
        context = self._context.build(chunks)

        # 3. Build messages (system + history + current query+context)
        prompt = build_rag_prompt(query, [b.text for b in context.blocks])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": prompt},
        ]

        # 4. LLM call
        response = await self._llm.complete(
            messages=messages,
            kind="smart",
            temperature=temperature,
        )

        # 5. Citations
        citations = build_citations(response.content, context)

        # 6. Persist
        conversation, message = await self._persist(
            db,
            query=query,
            answer=response.content,
            citations=citations,
            conversation_id=conversation_id,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=int((time.monotonic() - start) * 1000),
        )

        return RAGResult(
            answer=response.content,
            citations=citations,
            retrieved=chunks,
            conversation_id=conversation.id,
            message_id=message.id,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=int((time.monotonic() - start) * 1000),
        )

    # ─────────────────────────────────────────────
    # Streaming
    # ─────────────────────────────────────────────
    async def stream(
        self,
        db: AsyncSession,
        query: str,
        *,
        conversation_id: uuid.UUID | None = None,
        top_k: int | None = None,
        rerank_top_k: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[dict[str, object]]:
        """Stream a RAG answer as events."""
        start = time.monotonic()

        try:
            # 0. Load history
            history = await self._load_history(db, conversation_id)

            # 1. Retrieve
            chunks = await self._retrieval.search(
                db,
                query,
                top_k=top_k,
                rerank_top_k=rerank_top_k,
            )

            # 2. Context
            context = self._context.build(chunks)

            # 3. Emit citations first
            citations = [_chunk_to_citation_dict(c) for c in context.chunk_map.values()]
            yield {"type": "citations", "citations": citations}

            # 4. Stream LLM
            prompt = build_rag_prompt(query, [b.text for b in context.blocks])
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *history,
                {"role": "user", "content": prompt},
            ]

            answer_parts: list[str] = []
            async for delta in self._llm.stream(
                messages,
                kind="smart",
                temperature=temperature,
            ):
                answer_parts.append(delta)
                yield {"type": "token", "content": delta}

            answer = "".join(answer_parts)

            # 5. Final citations
            final_citations = build_citations(answer, context)

            # usage from last_usage
            usage = self._llm.last_usage or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))

            # 6. Persist
            conversation, message = await self._persist(
                db,
                query=query,
                answer=answer,
                citations=final_citations,
                conversation_id=conversation_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=int((time.monotonic() - start) * 1000),
            )

            yield {
                "type": "done",
                "conversation_id": str(conversation.id),
                "message_id": str(message.id),
                "latency_ms": int((time.monotonic() - start) * 1000),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

        except Exception as e:
            logger.exception("rag.stream_failed")
            yield {"type": "error", "error": str(e)}

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────
    async def _load_history(
        self,
        db: AsyncSession,
        conversation_id: uuid.UUID | None,
    ) -> list[dict[str, str]]:
        if not self._settings.rag_include_history:
            return []
        if conversation_id is None:
            return []

        max_n = self._settings.rag_max_history_messages
        if max_n <= 0:
            return []

        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .where(Message.role.in_(("user", "assistant")))
            .order_by(Message.created_at.desc())
            .limit(max_n)
        )
        result = await db.execute(stmt)
        rows = list(result.scalars().all())
        rows.reverse()  # oldest → newest

        # Token-budgeted history: keep dropping oldest until it fits
        budget = self._settings.rag_context_token_budget // 2  # half for history
        history: list[dict[str, str]] = []
        total = 0
        for msg in rows:
            tokens = self._llm.count_tokens(msg.content)
            if total + tokens > budget:
                break
            history.append({"role": msg.role, "content": msg.content})
            total += tokens

        logger.info(
            "rag.history_loaded",
            conversation_id=str(conversation_id),
            messages=len(history),
            tokens=total,
        )
        return history

    # ─────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────
    async def _persist(
        self,
        db: AsyncSession,
        *,
        query: str,
        answer: str,
        citations: list[Citation],
        conversation_id: uuid.UUID | None,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
    ) -> tuple[Conversation, Message]:
        """Save the conversation + messages to the DB."""
        # Get or create conversation
        conversation: Conversation | None = None
        if conversation_id is not None:
            conversation = await db.get(Conversation, conversation_id)

        if conversation is None:
            conversation = Conversation(
                id=uuid.uuid4(),
                title=query[:100],
                mode=self._settings.docu_mind_mode,
                llm_model=self._settings.active_llm_smart,
            )
            db.add(conversation)
            await db.flush()

        # User message
        user_msg = Message(
            id=uuid.uuid4(),
            conversation_id=conversation.id,
            role="user",
            content=query,
            citations=[],
        )
        db.add(user_msg)

        # Assistant message
        citations_json = [c.model_dump(mode="json") for c in citations]
        assistant_msg = Message(
            id=uuid.uuid4(),
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            citations=citations_json,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
        db.add(assistant_msg)

        await db.commit()
        await db.refresh(conversation)
        await db.refresh(assistant_msg)

        return conversation, assistant_msg


def _chunk_to_citation_dict(chunk: RetrievedChunk) -> dict[str, object]:
    """Convert RetrievedChunk to a citation dict (for streaming event)."""
    snippet = chunk.content[:200]
    if len(chunk.content) > 200:
        snippet += "..."
    return {
        "chunk_id": str(chunk.chunk_id),
        "document_id": str(chunk.document_id),
        "document_filename": chunk.document_filename,
        "page_number": chunk.page_number,
        "snippet": snippet,
        "score": chunk.score,
    }
