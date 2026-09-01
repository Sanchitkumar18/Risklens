"""Pydantic schemas for the alert engine."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class AlertThresholds(BaseModel):
    """Configurable risk limits. All limits are magnitudes / fractions.

    Defaults are chosen to be sensible and to exercise the engine on the demo book;
    tune per mandate.
    """

    var_limit: float = Field(default=0.02, description="Max acceptable 1-day VaR fraction.")
    volatility_limit: float = Field(default=0.20, description="Max annualized volatility.")
    drawdown_limit: float = Field(default=0.25, description="Max acceptable |drawdown|.")
    max_single_weight: float = Field(default=0.35, description="Max single-name weight.")
    anomaly_score_limit: float = Field(default=0.70, description="Anomaly-score alert threshold.")
    stress_loss_limit: float = Field(default=0.25, description="Max acceptable stress |loss|.")


class AlertRead(BaseModel):
    """A persisted alert."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    portfolio_id: int
    alert_type: str
    severity: str
    message: str
    metric_value: Decimal | None = None
    threshold: Decimal | None = None
    acknowledged: bool
    created_at: datetime | None = None


class AlertEvaluationResult(BaseModel):
    """Outcome of running the alert engine for a portfolio."""

    portfolio_id: int
    as_of_date: date | None
    breaches: int = Field(..., description="Total threshold breaches detected.")
    created: int = Field(..., description="New alerts persisted (after de-duplication).")
    existing: int = Field(..., description="Breaches that matched an existing alert.")
    by_severity: dict[str, int] = Field(default_factory=dict)
    alerts: list[AlertRead] = Field(default_factory=list)
