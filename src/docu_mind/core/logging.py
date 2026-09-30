"""Structured logging setup (structlog).

- Dev: pretty console output with colors.
- Prod: JSON output suitable for log aggregation.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from docu_mind.core.config import get_settings


def _drop_color_message_key(_logger: Any, _method_name: str, event_dict: EventDict) -> EventDict:
    """Remove duplicate `color_message` key from uvicorn logs."""
    event_dict.pop("color_message", None)
    return event_dict


def _add_app_context(_logger: Any, _method_name: str, event_dict: EventDict) -> EventDict:
    """Attach static app-level fields to every log record."""
    settings = get_settings()
    event_dict.setdefault("app", settings.app_name)
    event_dict.setdefault("env", settings.app_env)
    event_dict.setdefault("mode", settings.docu_mind_mode)
    return event_dict


def configure_logging() -> None:
    """Configure structlog + stdlib logging.

    Idempotent: safe to call multiple times.
    """
    settings = get_settings()
    level = getattr(logging, settings.app_log_level)

    # Shared processors for both dev and prod
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_app_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _drop_color_message_key,
    ]

    if settings.is_development:
        # Dev: pretty console with colors
        renderer: Processor = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # Prod: JSON
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Route stdlib logs (uvicorn, sqlalchemy, etc.) through structlog
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet down noisy libs
    for noisy in ("uvicorn.access", "uvicorn.error", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    # uvicorn loggers should propagate to root
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger.

    Args:
        name: Optional logger name (usually `__name__`).

    Returns:
        A structlog BoundLogger ready for structured logging.
    """
    return structlog.get_logger(name)  # type: ignore[no-any-return]
