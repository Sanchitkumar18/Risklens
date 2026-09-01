"""Market-data routes: CSV upload and per-ticker query."""

from __future__ import annotations

import io
from datetime import date

import pandas as pd
from fastapi import APIRouter, Depends, File, Query, UploadFile

from app.api.dependencies import get_market_data_service
from app.core.exceptions import DataValidationError
from app.schemas.market_data import IngestionSummary, MarketDataSeries
from app.services.market_data_service import MarketDataService

router = APIRouter(prefix="/market-data", tags=["market-data"])


@router.post("/upload", response_model=IngestionSummary, summary="Upload a market-data CSV")
async def upload_market_data(
    file: UploadFile = File(..., description="CSV with OHLCV columns."),
    service: MarketDataService = Depends(get_market_data_service),
) -> IngestionSummary:
    """Ingest a CSV of OHLCV bars through the validation pipeline (idempotent)."""
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:  # malformed CSV
        raise DataValidationError(
            "Could not parse the uploaded file as CSV.",
            details={"filename": file.filename},
        ) from exc
    return service.ingest_dataframe(df)


@router.get("/{ticker}", response_model=MarketDataSeries, summary="Get bars for a ticker")
def get_ticker(
    ticker: str,
    start: date | None = Query(default=None),
    end: date | None = Query(default=None),
    service: MarketDataService = Depends(get_market_data_service),
) -> MarketDataSeries:
    """Return stored OHLCV bars for a ticker over an optional date range."""
    return service.get_series(ticker, start=start, end=end)
