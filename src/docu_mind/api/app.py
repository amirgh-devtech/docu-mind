"""FastAPI application factory."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from docu_mind import __version__
from docu_mind.api.v1.router import api_router
from docu_mind.core.config import get_settings
from docu_mind.core.embeddings import EmbeddingClient
from docu_mind.core.llm import LLMClient
from docu_mind.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup / shutdown hooks."""
    settings = get_settings()
    configure_logging()

    # Propagate HF_TOKEN to HuggingFace Hub BEFORE any HF library is imported.
    if settings.hf_token:
        os.environ.setdefault("HF_TOKEN", settings.hf_token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", settings.hf_token)

    logger.info(
        "app.startup",
        version=__version__,
        env=settings.app_env,
        mode=settings.docu_mind_mode,
        hf_token_set=bool(settings.hf_token),
    )

    # Build shared clients
    app.state.llm_client = LLMClient(settings)
    app.state.embedding_client = EmbeddingClient(settings)

    # Pre-load the local reranker (moves the slow model load
    # out of the first user request path). Uses a module-level
    # cache so subsequent requests share the loaded model.
    if settings.active_reranker_provider == "local":
        from docu_mind.services.reranking.local import LocalReranker

        logger.info("app.preloading_reranker")
        reranker = LocalReranker(settings)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, reranker.preload)
        logger.info("app.reranker_ready")

    try:
        yield
    finally:
        logger.info("app.shutdown")
        await app.state.embedding_client.aclose()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="docu-mind",
        version=__version__,
        description=(
            "Personal document assistant with real RAG — "
            "upload PDF/Word/Markdown, ask questions, get cited answers."
        ),
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS — open in dev, tighten in prod
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(api_router)

    return app
