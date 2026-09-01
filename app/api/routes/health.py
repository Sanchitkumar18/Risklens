"""Health-check route.

A liveness probe that does not touch the database — used by Docker healthchecks,
load balancers, and the Streamlit dashboard to confirm the API is reachable. A
separate DB-aware readiness check is added in a later phase once the DB exists.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import __version__
from app.api.dependencies import get_db_session
from app.core.config import Settings, get_settings
from app.schemas.common import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health() -> HealthResponse:
    """Return basic service status. Always cheap; never touches external services."""
    settings: Settings = get_settings()
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        environment=settings.app_env,
        version=__version__,
    )


@router.get("/health/ready", response_model=ReadinessResponse, summary="Readiness probe")
def readiness(db: Session = Depends(get_db_session)) -> ReadinessResponse:
    """Confirm the database is reachable by issuing a trivial ``SELECT 1``.

    Used by orchestration to decide whether the service can serve traffic that
    depends on the datastore.
    """
    db.execute(text("SELECT 1"))
    return ReadinessResponse(status="ready", database="ok")
