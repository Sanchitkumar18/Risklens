"""Application configuration.

All runtime configuration is sourced from environment variables (12-factor style)
and validated through a single Pydantic ``Settings`` object. Import the cached
``get_settings()`` accessor rather than reading ``os.environ`` directly so that
configuration is validated once, typed, and easy to override in tests.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated application settings loaded from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ─────────────────────────────────────────
    app_name: str = "RiskLens"
    app_env: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    log_json: bool = False

    # ── Database ────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+psycopg2://risklens:risklens@localhost:5432/risklens",
    )

    # ── LLM / GenAI ─────────────────────────────────────────
    llm_provider: Literal["mock", "openai"] = "mock"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    model_name: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    # ── Risk engine ─────────────────────────────────────────
    default_confidence_level: float = 0.95
    trading_days_per_year: int = 252
    min_history_observations: int = 2

    # ── Caching ─────────────────────────────────────────────
    price_cache_ttl_seconds: int = 300

    # ── Validators ──────────────────────────────────────────
    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {value!r}")
        return upper

    @field_validator("default_confidence_level")
    @classmethod
    def _validate_confidence(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError(f"default_confidence_level must be in (0, 1), got {value}")
        return value

    @property
    def is_production(self) -> bool:
        """True when running in the production environment (controls error verbosity)."""
        return self.app_env == "production"

    @property
    def llm_enabled(self) -> bool:
        """True when a real LLM provider is configured with an API key."""
        return self.llm_provider == "openai" and bool(self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide, cached ``Settings`` instance.

    Cached so the ``.env`` file and environment are parsed exactly once. Tests can
    clear the cache with ``get_settings.cache_clear()`` after patching the env.
    """
    return Settings()
