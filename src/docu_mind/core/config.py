"""Application configuration (pydantic-settings)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Mode = Literal["eco", "pro"]


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "docu-mind"
    app_env: str = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"  # noqa: S104 — intentional: bind to all interfaces in containers
    app_port: int = 8000
    app_log_level: str = "INFO"

    # Mode
    docu_mind_mode: Mode = "eco"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
