"""LLM abstraction over LiteLLM.

Centralises:
- Model routing (fast vs smart, eco vs pro)
- Fallback chain
- Retry with backoff
- Token counting (heuristic, provider-agnostic)
- Streaming
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import os
from typing import Any, Literal

# Must be set BEFORE importing litellm so it skips the remote cost map fetch.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from docu_mind.core.config import Settings, get_settings
from docu_mind.core.exceptions import LLMProviderError, LLMTimeoutError
from docu_mind.core.logging import get_logger

logger = get_logger(__name__)

TaskKind = Literal["fast", "smart"]

# Quiet litellm's own logging — we handle it via structlog
litellm.suppress_debug_info = True
litellm.drop_params = True  # silently drop unsupported params per provider


# Heuristic constants for token estimation
_CHARS_PER_TOKEN_EN = 4  # ~4 chars per token for English/Latin scripts
_CJK_TOKEN_RATIO = 1  # ~1 char per token for CJK


@dataclass(slots=True)
class LLMResponse:
    """Normalised LLM response."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    """Async LLM client with routing, fallback, and retry."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # ─────────────────────────────────────────────
    # Token counting
    # ─────────────────────────────────────────────
    def count_tokens(self, text: str) -> int:
        """Estimate token count for a text string.

        We use a simple heuristic rather than a real tokenizer because:

        1. We use Gemini (and fallback providers), not OpenAI. Each provider
           has its own tokenizer, so pinning to `tiktoken`'s `o200k_base`
           would give misleading numbers anyway.
        2. `tiktoken` downloads its BPE files from OpenAI's CDN at runtime,
           which is unreachable from some regions (including Iran).

        The heuristic is:
        - CJK characters ≈ 1 token each
        - Everything else  ≈ 4 characters per token

        This is accurate enough for chunk sizing, which is the only
        place we use it. For exact counts (e.g. billing), call the
        provider's token counting API directly.

        Args:
            text: Input text.

        Returns:
            Estimated token count (>= 0).
        """
        if not text:
            return 0

        cjk = sum(
            1
            for c in text
            if "\u4e00" <= c <= "\u9fff"  # CJK Unified Ideographs
            or "\u3040" <= c <= "\u30ff"  # Hiragana + Katakana
            or "\uac00" <= c <= "\ud7af"  # Hangul syllables
        )
        non_cjk = len(text) - cjk
        return cjk + max(1, non_cjk // _CHARS_PER_TOKEN_EN)

    # ─────────────────────────────────────────────
    # Model selection
    # ─────────────────────────────────────────────
    def _models_for(self, kind: TaskKind) -> list[str]:
        """Return ordered list of models to try: primary + fallbacks.

        The fallback chain from settings is always appended (deduped).

        Args:
            kind: "fast" (chunking helpers) or "smart" (final answers).

        Returns:
            List of model identifiers in priority order.
        """
        primary = (
            self._settings.active_llm_fast if kind == "fast" else self._settings.active_llm_smart
        )
        chain = [primary]
        for m in self._settings.active_llm_fallback_chain:
            if m not in chain:
                chain.append(m)
        return chain

    # ─────────────────────────────────────────────
    # Core call
    # ─────────────────────────────────────────────
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        kind: TaskKind = "smart",
        temperature: float | None = None,
        max_tokens: int | None = None,
        model_override: str | None = None,
        **extra: Any,
    ) -> LLMResponse:
        """Run a chat completion with fallback chain + retry.

        Args:
            messages: OpenAI-style messages.
            kind: "fast" or "smart" — determines primary model.
            temperature: Overrides settings default.
            max_tokens: Overrides settings default.
            model_override: Force a specific model (skip routing).
            **extra: Forwarded to litellm.acompletion.

        Returns:
            Normalised LLMResponse.

        Raises:
            LLMProviderError: If all models in the chain fail.
            LLMTimeoutError: If a call times out.
        """
        models = [model_override] if model_override else self._models_for(kind)
        temp = self._settings.llm_temperature if temperature is None else temperature
        mtok = self._settings.llm_max_tokens if max_tokens is None else max_tokens

        last_error: Exception | None = None

        for model in models:
            try:
                return await self._call_with_retry(
                    model=model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=mtok,
                    **extra,
                )
            except (LLMProviderError, LLMTimeoutError) as e:
                last_error = e
                logger.warning(
                    "llm.attempt_failed",
                    model=model,
                    error=str(e),
                    kind=kind,
                )
                continue

        msg = f"All LLM models failed. Last error: {last_error}"
        raise LLMProviderError(msg) from last_error

    async def _call_with_retry(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **extra: Any,
    ) -> LLMResponse:
        """Single-model call with retry/backoff."""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._settings.llm_max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((TimeoutError, ConnectionError)),
            reraise=True,
        ):
            with attempt:
                try:
                    response = await litellm.acompletion(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=self._settings.llm_request_timeout,
                        **extra,
                    )
                except TimeoutError as e:
                    raise LLMTimeoutError(str(e)) from e
                except Exception as e:
                    raise LLMProviderError(f"{model}: {e}") from e

        return self._normalise(response, model)

    # ─────────────────────────────────────────────
    # Streaming
    # ─────────────────────────────────────────────
    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        kind: TaskKind = "smart",
        temperature: float | None = None,
        max_tokens: int | None = None,
        model_override: str | None = None,
        **extra: Any,
    ) -> AsyncIterator[str]:
        """Stream tokens from the LLM.

        Yields text deltas as they arrive. Falls back to next model if the
        first one fails before any token is produced.

        Args:
            messages: OpenAI-style messages.
            kind: "fast" or "smart".
            temperature: Override.
            max_tokens: Override.
            model_override: Force model.
            **extra: Forwarded to litellm.

        Yields:
            Text chunks (str).
        """
        models = [model_override] if model_override else self._models_for(kind)
        temp = self._settings.llm_temperature if temperature is None else temperature
        mtok = self._settings.llm_max_tokens if max_tokens is None else max_tokens

        last_error: Exception | None = None

        for model in models:
            try:
                stream = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=mtok,
                    timeout=self._settings.llm_request_timeout,
                    stream=True,
                    **extra,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
                return  # success — stop trying fallbacks
            except Exception as e:
                last_error = e
                logger.warning(
                    "llm.stream_failed",
                    model=model,
                    error=str(e),
                )
                continue

        msg = f"All LLM models failed (stream). Last error: {last_error}"
        raise LLMProviderError(msg) from last_error

    # ─────────────────────────────────────────────
    # Normalisation
    # ─────────────────────────────────────────────
    @staticmethod
    def _normalise(response: Any, model: str) -> LLMResponse:
        """Convert a litellm response into our LLMResponse."""
        choice = response.choices[0]
        usage = getattr(response, "usage", None)

        return LLMResponse(
            content=choice.message.content or "",
            model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
            finish_reason=getattr(choice, "finish_reason", None),
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )
