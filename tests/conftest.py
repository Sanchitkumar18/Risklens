"""Shared pytest fixtures.

Phase 1 keeps this minimal: an env-isolated ``TestClient`` for the FastAPI app.
Database fixtures are added in the persistence phase.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session", autouse=True)
def _test_env() -> None:
    """Force a deterministic, offline test environment before settings are read."""
    os.environ.setdefault("APP_ENV", "test")
    os.environ.setdefault("LLM_PROVIDER", "mock")
    os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """Return a FastAPI ``TestClient`` bound to a freshly built app."""
    from app.core.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()  # ensure test env is picked up
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
