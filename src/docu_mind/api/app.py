"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

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
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup / shutdown hooks."""
    settings = get_settings()
    configure_logging()
    logger.info(
        "app.startup",
        version=__version__,
        env=settings.app_env,
        mode=settings.docu_mind_mode,
    )

    # Lazily build clients and stash them on app.state
    app.state.llm_client = LLMClient(settings)
    app.state.embedding_client = EmbeddingClient(settings)

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
