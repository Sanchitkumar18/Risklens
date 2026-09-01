"""Risk routes: full risk report and correlation."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_risk_service
from app.schemas.risk import CorrelationResponse, RiskReport, RiskTimeSeriesPoint
from app.services.risk_service import RiskService

router = APIRouter(prefix="/portfolios", tags=["risk"])


@router.get("/{portfolio_id}/risk", response_model=RiskReport, summary="Compute risk report")
def get_risk(
    portfolio_id: int,
    confidence: float | None = Query(default=None, gt=0.0, lt=1.0),
    service: RiskService = Depends(get_risk_service),
) -> RiskReport:
    """Compute the full risk report (VaR, volatility, drawdown, contribution, …).

    Developer note: this persists a `risk_metrics` snapshot each call so the assistant
    can later explain *why risk changed* by diffing snapshots. That is a deliberate
    (small) write on a GET; a stricter API would move it behind POST.
    """
    return service.compute_metrics(portfolio_id, confidence_level=confidence, persist=True)


@router.get("/{portfolio_id}/correlation", response_model=CorrelationResponse)
def get_correlation(
    portfolio_id: int,
    service: RiskService = Depends(get_risk_service),
) -> CorrelationResponse:
    """Return the asset return correlation matrix and highly correlated pairs."""
    return service.compute_correlation(portfolio_id)


@router.get("/{portfolio_id}/risk/timeseries", response_model=list[RiskTimeSeriesPoint])
def get_risk_timeseries(
    portfolio_id: int,
    vol_window: int = Query(default=21, ge=2, le=252),
    service: RiskService = Depends(get_risk_service),
) -> list[RiskTimeSeriesPoint]:
    """Return portfolio value, drawdown, and rolling volatility over time (for charts)."""
    return service.compute_timeseries(portfolio_id, vol_window=vol_window)
