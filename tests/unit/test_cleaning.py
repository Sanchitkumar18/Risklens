"""Unit tests for the cleaning stage."""

from __future__ import annotations

import pandas as pd
import pytest

from app.pipelines.cleaning import clean_market_data, deduplicate, sort_bars


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-04", "2024-01-02", "2024-01-03"],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "open": [102.0, 100.0, 101.0],
            "high": [104.0, 102.0, 103.0],
            "low": [101.0, 99.0, 100.0],
            "close": [103.0, 101.0, 102.0],
            "adjusted_close": [103.0, 101.0, 102.0],
            "volume": [1200, 1000, 1100],
        }
    )


@pytest.mark.unit
def test_sort_bars_orders_by_ticker_date():
    out = sort_bars(_df())
    assert list(pd.to_datetime(out["date"]).dt.day) == [2, 3, 4]


@pytest.mark.unit
def test_deduplicate_keeps_last():
    df = _df()
    dup = df.iloc[[1]].copy()
    dup["close"] = 999.0  # a restated bar for the same (ticker, date)
    df2 = pd.concat([df, dup], ignore_index=True)

    out, removed = deduplicate(df2)
    assert removed == 1
    restated = out[(out["ticker"] == "AAPL") & (out["date"] == "2024-01-02")]
    assert restated.iloc[0]["close"] == 999.0


@pytest.mark.unit
def test_clean_reports_duplicates():
    df = _df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    out, stats = clean_market_data(df)
    assert stats.duplicates_removed == 1
    assert stats.rows_out == 3
    assert stats.notes  # a human-readable note was recorded


@pytest.mark.unit
def test_forward_fill_missing_business_days():
    # Skip 2024-01-03 (a Wednesday, business day).
    df = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-04"],
            "ticker": ["AAPL", "AAPL"],
            "open": [100.0, 102.0], "high": [102.0, 104.0], "low": [99.0, 101.0],
            "close": [101.0, 103.0], "adjusted_close": [101.0, 103.0],
            "volume": [1000, 1200],
        }
    )
    out, stats = clean_market_data(df, fill_missing_business_days=True)
    assert stats.rows_filled == 1
    filled = out[pd.to_datetime(out["date"]) == pd.Timestamp("2024-01-03")]
    assert len(filled) == 1
    assert filled.iloc[0]["volume"] == 0            # fills carry zero volume
    assert filled.iloc[0]["close"] == 101.0         # forward-filled from prior day
