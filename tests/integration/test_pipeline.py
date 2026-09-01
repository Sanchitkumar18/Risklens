"""Integration test: full ingestion pipeline (validate → clean → upsert)."""

from __future__ import annotations

import numpy as np
import pytest

from app.pipelines.synthetic_data import generate_market_data
from app.services.market_data_service import MarketDataService


@pytest.mark.integration
def test_pipeline_quarantines_bad_rows(db_session):
    df = generate_market_data(start="2022-01-03", end="2022-02-28", seed=11)
    total = len(df)

    # Corrupt exactly two rows in ways that must be rejected.
    df.loc[0, "close"] = -1.0          # non-positive price
    df.loc[5, "high"] = 0.01           # high < low
    df.loc[5, "low"] = 500.0

    service = MarketDataService(db_session)
    summary = service.ingest_dataframe(df)

    assert summary.validation is not None
    assert summary.rows_rejected == 2
    assert summary.rows_written == total - 2
    assert summary.rows_read == total

    # Only accepted rows reached the database.
    stored = len(service.repo.get_for_tickers(list(df["ticker"].unique())))
    assert stored == total - 2


@pytest.mark.integration
def test_pipeline_clean_data_all_written(db_session):
    df = generate_market_data(start="2022-01-03", end="2022-01-31", seed=1)
    service = MarketDataService(db_session)
    summary = service.ingest_dataframe(df)

    assert summary.rows_rejected == 0
    assert summary.validation.is_valid
    assert summary.rows_written == len(df)


@pytest.mark.integration
def test_pipeline_deduplicates_before_write(db_session):
    import pandas as pd

    df = generate_market_data(start="2022-01-03", end="2022-01-31", seed=1)
    # Duplicate the whole frame → validation flags duplicates, cleaning removes them.
    combined = pd.concat([df, df], ignore_index=True)

    service = MarketDataService(db_session)
    summary = service.ingest_dataframe(combined)

    # Written row count equals the distinct (ticker, date) count, not the doubled input.
    assert summary.rows_written == len(df)
