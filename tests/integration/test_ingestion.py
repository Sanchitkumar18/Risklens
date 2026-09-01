"""Integration tests for the ingestion pipeline + market-data service."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from app.core.exceptions import DataValidationError
from app.pipelines.ingestion import dataframe_to_rows, load_csv, normalize_columns
from app.pipelines.synthetic_data import generate_market_data
from app.services.market_data_service import MarketDataService


def _small_df() -> pd.DataFrame:
    return generate_market_data(start="2022-01-03", end="2022-02-28", seed=3)


# ── pure parsing ────────────────────────────────────────────
@pytest.mark.integration
def test_dataframe_to_rows_types():
    df = _small_df().head(3)
    rows = dataframe_to_rows(df)
    assert len(rows) == 3
    row = rows[0]
    assert isinstance(row["date"], date)
    assert isinstance(row["open"], Decimal)
    assert isinstance(row["volume"], int)
    assert row["ticker"].isupper()


@pytest.mark.integration
def test_column_alias_normalization():
    df = pd.DataFrame(
        {
            "Date": ["2022-01-03"],
            "Symbol": ["aapl"],
            "Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.5],
            "Adj Close": [100.5], "Volume": [1000],
        }
    )
    norm = normalize_columns(df)
    assert "adjusted_close" in norm.columns
    assert "ticker" in norm.columns
    rows = dataframe_to_rows(df)
    assert rows[0]["ticker"] == "AAPL"


@pytest.mark.integration
def test_missing_column_raises():
    df = pd.DataFrame({"date": ["2022-01-03"], "ticker": ["AAPL"], "close": [100.0]})
    with pytest.raises(DataValidationError):
        dataframe_to_rows(df)


@pytest.mark.integration
def test_unparseable_date_raises():
    df = pd.DataFrame(
        {
            "date": ["not-a-date"],
            "ticker": ["AAPL"],
            "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5],
            "adjusted_close": [100.5], "volume": [1000],
        }
    )
    with pytest.raises(DataValidationError):
        dataframe_to_rows(df)


# ── service → database ──────────────────────────────────────
@pytest.mark.integration
def test_ingest_dataframe_persists(db_session):
    df = _small_df()
    service = MarketDataService(db_session)
    summary = service.ingest_dataframe(df)

    assert summary.rows_written == len(df)
    assert summary.rows_read == len(df)
    assert set(summary.tickers) == set(df["ticker"].unique())
    assert service.repo.count_for_ticker("AAPL") == len(df[df["ticker"] == "AAPL"])


@pytest.mark.integration
def test_ingest_is_idempotent(db_session):
    df = _small_df()
    service = MarketDataService(db_session)
    service.ingest_dataframe(df)
    first_total = len(service.repo.get_for_tickers(list(df["ticker"].unique())))

    service.ingest_dataframe(df)  # re-ingest same rows
    second_total = len(service.repo.get_for_tickers(list(df["ticker"].unique())))
    assert first_total == second_total == len(df)


@pytest.mark.integration
def test_csv_round_trip(tmp_path, db_session):
    df = _small_df()
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_csv(csv_path)
    assert "adjusted_close" in loaded.columns

    service = MarketDataService(db_session)
    summary = service.ingest_csv(str(csv_path))
    assert summary.rows_written == len(df)


@pytest.mark.integration
def test_missing_csv_raises():
    with pytest.raises(DataValidationError):
        load_csv("does/not/exist.csv")
