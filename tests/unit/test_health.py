"""Unit tests for the health route, config, and exception hierarchy (Phase 1)."""

from __future__ import annotations

import pytest

from app import __version__
from app.core.exceptions import (
    PortfolioNotFound,
    RiskCalculationError,
    RiskLensError,
)


@pytest.mark.unit
def test_health_endpoint_returns_ok(client) -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "ok"
    assert body["app"] == "RiskLens"
    assert body["environment"] == "test"
    assert body["version"] == __version__


@pytest.mark.unit
def test_health_sets_request_id_header(client) -> None:
    resp = client.get("/api/v1/health")
    assert "X-Request-ID" in resp.headers
    assert len(resp.headers["X-Request-ID"]) > 0


@pytest.mark.unit
def test_unknown_route_returns_404(client) -> None:
    resp = client.get("/api/v1/does-not-exist")
    assert resp.status_code == 404


@pytest.mark.unit
def test_exception_hierarchy_and_envelope() -> None:
    exc = PortfolioNotFound("Portfolio 7 not found", details={"portfolio_id": 7})

    assert isinstance(exc, RiskLensError)
    assert exc.code == "PORTFOLIO_NOT_FOUND"
    assert exc.http_status == 404
    assert exc.to_dict() == {
        "code": "PORTFOLIO_NOT_FOUND",
        "message": "Portfolio 7 not found",
        "details": {"portfolio_id": 7},
    }


@pytest.mark.unit
def test_risk_calculation_error_is_5xx() -> None:
    exc = RiskCalculationError("covariance matrix is singular")
    assert exc.http_status == 500
    assert exc.code == "RISK_CALCULATION_ERROR"


@pytest.mark.unit
def test_settings_validation_rejects_bad_confidence(monkeypatch) -> None:
    from app.core.config import Settings

    monkeypatch.setenv("DEFAULT_CONFIDENCE_LEVEL", "1.5")
    with pytest.raises(ValueError):
        Settings()
