"""Shared pytest fixtures.

Database tests run against an **in-memory SQLite** database built from the ORM
metadata, so the suite is fully self-contained (no Docker/Postgres needed). The
models are written to be dialect-portable; the Alembic migration is separately
verified against SQLite in the Phase 2 checks and runs on Postgres in production.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture(scope="session", autouse=True)
def _test_env() -> None:
    """Force a deterministic, offline test environment before settings are read."""
    os.environ.setdefault("APP_ENV", "test")
    os.environ.setdefault("LLM_PROVIDER", "mock")
    os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture()
def db_engine() -> Iterator[Engine]:
    """Create a fresh in-memory SQLite engine with the full schema."""
    from app.db import models  # noqa: F401  (register tables on metadata)
    from app.db.database import Base

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _enable_fk(dbapi_conn, _record) -> None:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture()
def db_session(db_engine: Engine) -> Iterator[Session]:
    """Yield a session bound to the test engine, committing within the test."""
    factory = sessionmaker(bind=db_engine, autoflush=False, expire_on_commit=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """FastAPI ``TestClient`` for routes that don't need a database."""
    from app.core.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def client_db(db_engine: Engine) -> Iterator[TestClient]:
    """FastAPI ``TestClient`` with the DB dependency overridden to the test engine."""
    from app.api.dependencies import get_db_session
    from app.core.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    app = create_app()

    factory = sessionmaker(bind=db_engine, autoflush=False, expire_on_commit=False)

    def _override() -> Iterator[Session]:
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db_session] = _override
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
