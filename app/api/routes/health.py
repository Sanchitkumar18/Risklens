"""Health-check route.

A liveness probe that does not touch the database — used by Docker healthchecks,
load balancers, and the Streamlit dashboard to confirm the API is reachable. A
separate DB-aware readiness check is added in a later phase once the DB exists.
"""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.core.config import Settings, get_settings
from app.schemas.common import HealthResponse

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
