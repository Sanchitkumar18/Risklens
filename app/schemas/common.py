"""Shared Pydantic schemas used across multiple API routes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Machine + human readable error body nested under the ``error`` envelope."""

    code: str = Field(..., description="Stable, machine-readable error code.")
    message: str = Field(..., description="Human-readable error description.")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Optional structured context (never secrets)."
    )


class ErrorResponse(BaseModel):
    """Standard error envelope returned by every failed request."""

    error: ErrorDetail


class HealthResponse(BaseModel):
    """Response body for ``GET /api/v1/health``."""

    status: str = Field(..., description="'ok' when the service is healthy.")
    app: str = Field(..., description="Application name.")
    environment: str = Field(..., description="Deployment environment.")
    version: str = Field(..., description="Application version.")


class ReadinessResponse(BaseModel):
    """Response body for ``GET /api/v1/health/ready``."""

    status: str = Field(..., description="'ready' when dependencies are reachable.")
    database: str = Field(..., description="'ok' when the database responds.")
