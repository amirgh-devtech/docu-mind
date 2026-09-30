"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from docu_mind import __version__


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance.
    """
    return FastAPI(
        title="docu-mind",
        version=__version__,
        description="Personal document assistant with real RAG.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
