"""Pydantic schemas for anomaly detection."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class AnomalyRead(BaseModel):
    """A persisted anomaly row."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    portfolio_id: int | None = None
    ticker: str
    date: date
    anomaly_score: float
    anomaly_type: str
    features: dict | None = None
    created_at: datetime | None = None


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
