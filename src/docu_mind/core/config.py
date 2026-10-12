"""Application configuration (pydantic-settings).

Loads from environment / .env file. Provides a cached `get_settings()`.

The `docu_mind_mode` setting (eco | pro) is the central switch that
determines which LLM, embedding model, and reranker are used across
the entire pipeline.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Mode = Literal["eco", "pro"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─────────────────────────────────────────────
    # App
    # ─────────────────────────────────────────────
    app_name: str = "docu-mind"
    app_env: Literal["development", "staging", "production"] = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"  # noqa: S104 — intentional: bind all in container
    app_port: int = Field(default=8000, ge=1, le=65535)
    app_log_level: LogLevel = "INFO"

    # ─────────────────────────────────────────────
    # Mode
    # ─────────────────────────────────────────────
    docu_mind_mode: Mode = "eco"

    # ─────────────────────────────────────────────
    # LLM providers
    # ─────────────────────────────────────────────
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    groq_api_key: str | None = None

    # ─────────────────────────────────────────────
    # Per-task LLM routing
    # ─────────────────────────────────────────────
    # "fast" tasks: chunking context, title generation, etc.
    # "smart" tasks: final RAG answer, complex reasoning.
    llm_fast_eco: str = "gemini/gemini-3.5-flash-lite"
    llm_smart_eco: str = "gemini/gemini-3.5-flash-lite"
    llm_fast_pro: str = "gemini/gemini-3.5-flash-lite"
    llm_smart_pro: str = "gemini/gemini-3.6-flash"

    # Fallback chain (used if primary fails, in order)
    llm_fallback_chain_eco: list[str] = [
        "gemini/gemini-3.5-flash-lite",
        "groq/llama-3.3-70b-versatile",
    ]
    llm_fallback_chain_pro: list[str] = [
        "gemini/gemini-3.6-flash",
        "openai/gpt-5.4-mini",
        "anthropic/claude-haiku-4.5",
    ]

    # LLM runtime
    llm_request_timeout: int = Field(default=180, ge=5, le=600)
    llm_max_retries: int = Field(default=3, ge=0, le=10)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, ge=1, le=32000)

    # ─────────────────────────────────────────────
    # Embedding
    # ─────────────────────────────────────────────
    embedding_provider: Literal["openai", "gemini"] = "gemini"

    # Eco mode
    embedding_model_eco_gemini: str = "gemini-embedding-001"
    embedding_model_eco_openai: str = "text-embedding-3-small"
    embedding_dim_eco: int = 1536

    # Pro mode
    embedding_model_pro_gemini: str = "gemini-embedding-001"
    embedding_model_pro_openai: str = "text-embedding-3-large"
    embedding_dim_pro: int = 1536

    # Common
    embedding_batch_size: int = Field(default=100, ge=1, le=2048)

    # ─────────────────────────────────────────────
    # Reranker
    # ─────────────────────────────────────────────
    reranker_provider_eco: Literal["local"] = "local"
    reranker_provider_pro: Literal["cohere"] = "cohere"
    reranker_model_local: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_model_pro: str = "rerank-v3.5"
    cohere_api_key: str | None = None

    # ─────────────────────────────────────────────
    # Postgres
    # ─────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_db: str = "docu_mind"
    postgres_user: str = "docu_mind"
    postgres_password: str = "change_me_in_prod"  # noqa: S105 — dev default, must be overridden
    database_url: str | None = None  # overrides the above if set

    # ─────────────────────────────────────────────
    # Redis
    # ─────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = Field(default=6379, ge=1, le=65535)
    redis_db: int = Field(default=0, ge=0, le=15)
    redis_url: str | None = None  # overrides the above if set

    # ─────────────────────────────────────────────
    # Embedding cache
    # ─────────────────────────────────────────────
    embedding_cache_backend: Literal["memory", "redis", "none"] = "redis"
    embedding_cache_ttl: int = Field(default=3600, ge=0, le=86400)

    # ─────────────────────────────────────────────
    # Chunking
    # ─────────────────────────────────────────────
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    contextual_chunking: bool = True

    # ─────────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────────
    retrieval_top_k: int = Field(default=20, ge=1, le=100)
    rerank_top_k: int = Field(default=5, ge=1, le=50)
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1, le=200)

    #: RRF weights for hybrid retrieval.
    #: Dense (semantic) is typically more reliable than keyword,
    #: so it gets a higher weight by default.
    rrf_dense_weight: float = Field(default=0.7, ge=0.0, le=2.0)
    rrf_keyword_weight: float = Field(default=0.3, ge=0.0, le=2.0)

    # ─────────────────────────────────────────────
    # Retrieval — keyword search
    # ─────────────────────────────────────────────
    #: PostgreSQL text search config. Options:
    #:   - "simple": no stemming, no stopwords, language-agnostic (default)
    #:   - "english": English stemming + stopwords
    #:   - "french", "german", ...: language-specific
    #: Use "simple" for multi-lingual documents; "english" (etc.) for
    #: single-language corpora where stemming helps recall.
    keyword_ts_config: str = "english"

    #: Query mode. Options:
    #:   - "websearch": Google-like OR semantics + phrase support (default)
    #:   - "plain": strict AND semantics (original behavior)
    #:   - "phrase": phrase search
    keyword_query_mode: Literal["websearch", "plain", "phrase"] = "websearch"

    # ─────────────────────────────────────────────
    # RAG
    # ─────────────────────────────────────────────
    rag_context_token_budget: int = Field(default=6000, ge=500, le=100000)

    #: Maximum number of prior messages (user+assistant) to include.
    #: 0 = no history (stateless).
    rag_max_history_messages: int = Field(default=10, ge=0, le=50)

    #: Whether to include the conversation history in the LLM call.
    rag_include_history: bool = True

    # ─────────────────────────────────────────────
    # Uploads
    # ─────────────────────────────────────────────
    upload_dir: Path = Path("./uploads")
    max_upload_size_mb: int = Field(default=50, ge=1, le=500)

    # ─────────────────────────────────────────────
    # HuggingFace
    # ─────────────────────────────────────────────
    hf_token: str | None = None

    # ─────────────────────────────────────────────
    # Security
    # ─────────────────────────────────────────────
    secret_key: str = "change_me_to_a_long_random_string"  # noqa: S105 — dev default, must be overridden

    # ─────────────────────────────────────────────
    # Validators
    # ─────────────────────────────────────────────
    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_lt_size(cls, v: int, info: object) -> int:
        # Note: pydantic v2 gives us the partially-built model via info.data
        data = getattr(info, "data", {})
        size = data.get("chunk_size", 800)
        if v >= size:
            msg = f"chunk_overlap ({v}) must be smaller than chunk_size ({size})"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _resolve_database_url(self) -> Settings:
        if not self.database_url:
            self.database_url = (
                f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
        return self

    @model_validator(mode="after")
    def _resolve_redis_url(self) -> Settings:
        if not self.redis_url:
            self.redis_url = f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return self

    @model_validator(mode="after")
    def _check_required_keys(self) -> Settings:
        """Validate that the keys needed for the current mode are present."""
        # ── LLM providers ──
        if self.docu_mind_mode == "eco":
            if not any([self.gemini_api_key, self.groq_api_key, self.openai_api_key]):
                msg = (
                    "Eco mode requires at least one of: "
                    "GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY"
                )
                raise ValueError(msg)
        elif not any(
            [
                self.gemini_api_key,
                self.openai_api_key,
                self.anthropic_api_key,
            ]
        ):  # pro
            msg = (
                "Pro mode requires at least one of: "
                "GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY"
            )
            raise ValueError(msg)

        # ── Embedding provider-specific keys ──
        if self.embedding_provider == "openai" and not self.openai_api_key:
            msg = "EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY"
            raise ValueError(msg)

        if self.embedding_provider == "gemini" and not self.gemini_api_key:
            msg = "EMBEDDING_PROVIDER=gemini requires GEMINI_API_KEY"
            raise ValueError(msg)

        # ── Pro-mode: Cohere reranker ──
        if (
            self.docu_mind_mode == "pro"
            and self.reranker_provider_pro == "cohere"
            and not self.cohere_api_key
        ):
            msg = "Pro mode with Cohere reranker requires COHERE_API_KEY"
            raise ValueError(msg)

        return self

    # ─────────────────────────────────────────────
    # Convenience properties
    # ─────────────────────────────────────────────
    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def active_llm_fast(self) -> str:
        return self.llm_fast_eco if self.docu_mind_mode == "eco" else self.llm_fast_pro

    @property
    def active_llm_smart(self) -> str:
        return self.llm_smart_eco if self.docu_mind_mode == "eco" else self.llm_smart_pro

    @property
    def active_llm_fallback_chain(self) -> list[str]:
        if self.docu_mind_mode == "eco":
            return self.llm_fallback_chain_eco
        return self.llm_fallback_chain_pro

    @property
    def active_embedding_model(self) -> str:
        if self.embedding_provider == "gemini":
            return (
                self.embedding_model_eco_gemini
                if self.docu_mind_mode == "eco"
                else self.embedding_model_pro_gemini
            )
        return (
            self.embedding_model_eco_openai
            if self.docu_mind_mode == "eco"
            else self.embedding_model_pro_openai
        )

    @property
    def active_embedding_dim(self) -> int:
        return self.embedding_dim_eco if self.docu_mind_mode == "eco" else self.embedding_dim_pro

    @property
    def active_reranker_provider(self) -> str:
        return (
            self.reranker_provider_eco
            if self.docu_mind_mode == "eco"
            else self.reranker_provider_pro
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings.

    The cache is cleared on process restart. In tests, use
    `get_settings.cache_clear()` to reset.
    """
    return Settings()
