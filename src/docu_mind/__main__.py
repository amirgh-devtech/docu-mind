"""Entrypoint: `python -m docu_mind`."""

from __future__ import annotations


def main() -> None:
    """Run the docu-mind API server."""
    import uvicorn

    from docu_mind.core.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "docu_mind.api.app:create_app",
        factory=True,
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
        log_level=settings.app_log_level.lower(),
    )


if __name__ == "__main__":
    main()
