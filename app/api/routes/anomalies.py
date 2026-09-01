"""Anomaly-detection routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_anomaly_service
from app.schemas.anomaly import AnomalyRead, AnomalyScanResult
from app.services.anomaly_service import AnomalyService

router = APIRouter(prefix="/portfolios", tags=["anomalies"])


@router.get("/{portfolio_id}/anomalies", response_model=list[AnomalyRead])
def list_anomalies(
    portfolio_id: int,
    limit: int | None = Query(default=100, ge=1, le=1000),
    service: AnomalyService = Depends(get_anomaly_service),
) -> list[AnomalyRead]:
    """List previously detected anomalies for a portfolio."""
    return service.list_persisted(portfolio_id, limit=limit)


@router.post("/{portfolio_id}/anomalies/scan", response_model=AnomalyScanResult)
def scan_anomalies(
    portfolio_id: int,
    contamination: float = Query(default=0.02, gt=0.0, lt=0.5),
    service: AnomalyService = Depends(get_anomaly_service),
) -> AnomalyScanResult:
    """Run Isolation Forest anomaly detection over the portfolio's tickers and persist."""
    return service.scan_portfolio(portfolio_id, contamination=contamination)
