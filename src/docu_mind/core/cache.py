"""Cache abstraction over Redis + in-memory backends.

Used for:
    - Query embedding cache (avoid re-embedding identical queries)

Design:
    - ``CacheBackend`` protocol defines the minimal interface.
    - ``MemoryCache`` uses ``cachetools.TTLCache`` (single-process).
    - ``RedisCache`` uses ``redis.asyncio`` (shared across workers).
    - ``build_cache()`` picks the backend from settings.

All backends are async-friendly. Redis operations are serialized with
``orjson`` for speed.
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

from cachetools import TTLCache
import orjson

from docu_mind.core.config import Settings, get_settings
from docu_mind.core.logging import get_logger

logger = get_logger(__name__)


class CacheBackend(Protocol):
    """Minimal cache protocol."""

    async def get(self, key: str) -> Any | None:
        """Return the cached value or None."""
        ...

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value with optional TTL (seconds)."""
        ...

    async def delete(self, key: str) -> None:
        """Delete a key if it exists."""
        ...

    async def close(self) -> None:
        """Release backend resources."""
        ...


# ─────────────────────────────────────────────
# In-memory backend
# ─────────────────────────────────────────────
class MemoryCache:
    """In-process TTL cache. Not shared across workers."""

    def __init__(self, maxsize: int = 512, ttl: int = 3600) -> None:
        self._cache: TTLCache[str, Any] = TTLCache(maxsize=maxsize, ttl=ttl)
        self._default_ttl = ttl

    async def get(self, key: str) -> Any | None:
        return self._cache.get(key)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        # MemoryCache uses a process-wide TTL; per-call ttl is ignored.
        _ = ttl
        self._cache[key] = value

    async def delete(self, key: str) -> None:
        self._cache.pop(key, None)

    async def close(self) -> None:
        self._cache.clear()


# ─────────────────────────────────────────────
# Redis backend
# ─────────────────────────────────────────────
class RedisCache:
    """Redis-backed cache. Shared across all workers."""

    def __init__(self, url: str, default_ttl: int = 3600) -> None:
        # Imported here so that the module can be imported even when
        # redis isn't installed (e.g. in tests with memory backend).
        import redis.asyncio as aioredis

        self._client = aioredis.from_url(
            url,
            encoding="utf-8",
            decode_responses=False,
        )
        self._default_ttl = default_ttl

    async def get(self, key: str) -> Any | None:
        try:
            raw = await self._client.get(key)
        except Exception as e:
            logger.warning("cache.redis_get_failed", key=key, error=str(e))
            return None
        if raw is None:
            return None
        try:
            return orjson.loads(raw)
        except orjson.JSONDecodeError:
            logger.warning("cache.redis_decode_failed", key=key)
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        try:
            raw = orjson.dumps(value)
            if effective_ttl > 0:
                await self._client.setex(key, effective_ttl, raw)
            else:
                await self._client.set(key, raw)
        except Exception as e:
            logger.warning("cache.redis_set_failed", key=key, error=str(e))

    async def delete(self, key: str) -> None:
        try:
            await self._client.delete(key)
        except Exception as e:
            logger.warning("cache.redis_delete_failed", key=key, error=str(e))

    async def close(self) -> None:
        try:
            await self._client.aclose()
        except Exception as e:
            logger.warning("cache.redis_close_failed", error=str(e))


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────
def build_cache(settings: Settings | None = None) -> CacheBackend | None:
    """Build the appropriate cache backend from settings.

    Returns ``None`` if the backend is disabled, or a MemoryCache
    fallback if the Redis backend can't be initialized.

    Raises:
        ValueError: If ``settings.embedding_cache_backend`` is an unknown
            value (should not happen with the Literal type, but defensive).
    """
    settings = settings or get_settings()
    backend = settings.embedding_cache_backend

    if backend == "none":
        logger.info("cache.disabled")
        return None

    if backend == "memory":
        logger.info("cache.backend", backend="memory")
        return MemoryCache(ttl=settings.embedding_cache_ttl)

    # backend == "redis"
    if not settings.redis_url:
        logger.warning("cache.redis_url_missing_falling_back_to_memory")
        return MemoryCache(ttl=settings.embedding_cache_ttl)
    try:
        logger.info("cache.backend", backend="redis", url=settings.redis_url)
        return RedisCache(
            url=settings.redis_url,
            default_ttl=settings.embedding_cache_ttl,
        )
    except Exception as e:
        logger.warning(
            "cache.redis_init_failed_falling_back_to_memory",
            error=str(e),
        )
        return MemoryCache(ttl=settings.embedding_cache_ttl)


# ─────────────────────────────────────────────
# Cache key helper (shared by embeddings and future consumers)
# ─────────────────────────────────────────────
def make_cache_key(*parts: str, prefix: str = "docu-mind") -> str:
    """Build a deterministic, namespaced cache key from string parts.

    Example:
        make_cache_key("embed", "query", "gemini-embedding-001", "hello")
        → "docu-mind:embed:query:gemini-embedding-001:2cf24dba..."
    """
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}:{':'.join(parts)}:{digest}"
