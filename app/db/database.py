"""Database engine, session factory, and the declarative ``Base``.

The engine is built lazily from ``Settings.database_url`` so that importing this
module never opens a connection (important for unit tests that don't need a DB).

Portability note: production runs on PostgreSQL, but the models and this engine are
written to also work on SQLite so the test suite can run fully in-memory without a
running database. SQLite-specific concerns (thread checks, foreign-key enforcement)
are handled here.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model."""


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _build_engine(url: str) -> Engine:
    """Create an :class:`Engine` with dialect-appropriate options."""
    if url.startswith("sqlite"):
        # In-memory / file SQLite: allow cross-thread use and keep a single
        # connection alive so an in-memory DB survives between sessions.
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool if ":memory:" in url or url == "sqlite://" else None,
        )

        @event.listens_for(engine, "connect")
        def _enable_sqlite_fk(dbapi_conn, _record) -> None:  # pragma: no cover - trivial
            # SQLite does not enforce foreign keys unless explicitly enabled.
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine

    # PostgreSQL (production): pre-ping avoids handing out dead connections.
    return create_engine(url, pool_pre_ping=True, future=True)


def _init() -> None:
    global _engine, _SessionLocal
    if _engine is None:
        settings = get_settings()
        _engine = _build_engine(settings.database_url)
        _SessionLocal = sessionmaker(
            bind=_engine, autocommit=False, autoflush=False, expire_on_commit=False
        )


def get_engine() -> Engine:
    """Return the process-wide engine, building it on first use."""
    _init()
    assert _engine is not None
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide session factory, building it on first use."""
    _init()
    assert _SessionLocal is not None
    return _SessionLocal


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a transactional session, closed on exit.

    The caller (service/repository) is responsible for committing; on an unhandled
    exception the session is rolled back before being closed.
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Dispose and clear the cached engine/session factory (used by tests)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
