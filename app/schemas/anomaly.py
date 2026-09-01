"""Pydantic schemas for anomaly detection."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class AnomalyRecord(BaseModel):
    """A single detected anomaly."""

    ticker: str
    date: date
    anomaly_score: float = Field(..., description="Higher = more anomalous.")
    anomaly_type: str = Field(..., description="Dominant deviation category.")
    features: dict[str, float] = Field(default_factory=dict)


class AnomalyScanResult(BaseModel):
    """Outcome of an anomaly-detection scan."""

    portfolio_id: int | None = None
    tickers: list[str] = Field(default_factory=list)
    rows_analyzed: int
    anomalies_found: int
    contamination: float
    anomalies: list[AnomalyRecord] = Field(default_factory=list)
