"""Alert routes: evaluate thresholds, list, and acknowledge."""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Query

from app.api.dependencies import get_alert_service
from app.schemas.alert import AlertEvaluationResult, AlertRead, AlertThresholds
from app.services.alert_service import AlertService

router = APIRouter(prefix="/portfolios", tags=["alerts"])


@router.get("/{portfolio_id}/alerts", response_model=list[AlertRead])
def list_alerts(
    portfolio_id: int,
    acknowledged: bool | None = Query(default=None),
    service: AlertService = Depends(get_alert_service),
) -> list[AlertRead]:
    """List a portfolio's alerts, optionally filtered by acknowledgement status."""
    return service.list_alerts(portfolio_id, acknowledged=acknowledged)


@router.post("/{portfolio_id}/alerts/scan", response_model=AlertEvaluationResult)
def scan_alerts(
    portfolio_id: int,
    thresholds: AlertThresholds | None = Body(default=None),
    service: AlertService = Depends(get_alert_service),
) -> AlertEvaluationResult:
    """Evaluate risk thresholds and persist any new alerts (de-duplicated)."""
    return service.evaluate(portfolio_id, thresholds)


@router.post("/{portfolio_id}/alerts/{alert_id}/acknowledge", response_model=AlertRead)
def acknowledge_alert(
    portfolio_id: int,
    alert_id: int,
    service: AlertService = Depends(get_alert_service),
) -> AlertRead:
    """Mark an alert as acknowledged."""
    return service.acknowledge(portfolio_id, alert_id)
