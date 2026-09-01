"""Pydantic schemas for market-data endpoints and ingestion results."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.validation import ValidationReport


class IngestionSummary(BaseModel):
    """Outcome of an ingestion run (returned by upload/seed operations)."""

    rows_read: int = Field(..., description="Rows parsed from the source.")
    rows_written: int = Field(..., description="Rows inserted or updated in the database.")
    rows_rejected: int = Field(default=0, description="Rows dropped by validation.")
    tickers: list[str] = Field(default_factory=list, description="Distinct tickers ingested.")
    validation: ValidationReport | None = Field(
        default=None, description="Validation report, if validation ran."
    )
    message: str = Field(default="", description="Human-readable summary.")


class MarketDataBar(BaseModel):
    """A single OHLCV bar as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    ticker: str
    date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    adjusted_close: Decimal
    volume: int
    created_at: datetime | None = None


class MarketDataSeries(BaseModel):
    """A ticker's bars over a range."""

    ticker: str
    count: int
    bars: list[MarketDataBar]
