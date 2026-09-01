"""Unit tests for the market-data validation stage."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.pipelines.validation import (
    CAT_ABNORMAL_JUMP,
    CAT_DUPLICATE_ROW,
    CAT_FUTURE_DATE,
    CAT_HIGH_LT_LOW,
    CAT_INVALID_TICKER,
    CAT_MISSING_DATE,
    CAT_NON_POSITIVE_PRICE,
    CAT_NULL_PRICE,
    CAT_OHLC_INCONSISTENT,
    validate_market_data,
)

REF = date(2025, 1, 1)


def good_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "adjusted_close": [101.0, 102.0, 103.0],
            "volume": [1000, 1100, 1200],
        }
    )


@pytest.mark.unit
def test_clean_data_passes():
    res = validate_market_data(good_df(), reference_date=REF)
    assert res.report.is_valid
    assert res.report.rows_rejected == 0
    assert res.report.rows_accepted == 3
    assert len(res.accepted) == 3
    assert res.rejected.empty


@pytest.mark.unit
def test_null_price_rejected():
    df = good_df()
    df.loc[1, "close"] = np.nan
    res = validate_market_data(df, reference_date=REF)
    assert res.report.rows_rejected == 1
    assert CAT_NULL_PRICE in res.report.errors_by_category
    assert len(res.accepted) == 2


@pytest.mark.unit
def test_negative_price_rejected():
    df = good_df()
    df.loc[0, "open"] = -5.0
    res = validate_market_data(df, reference_date=REF)
    assert CAT_NON_POSITIVE_PRICE in res.report.errors_by_category
    assert res.report.rows_rejected == 1


@pytest.mark.unit
def test_high_lt_low_rejected():
    df = good_df()
    df.loc[2, "high"] = 90.0  # below low (101)
    res = validate_market_data(df, reference_date=REF)
    assert CAT_HIGH_LT_LOW in res.report.errors_by_category


@pytest.mark.unit
def test_close_outside_ohlc_rejected():
    df = good_df()
    df.loc[0, "close"] = 500.0  # above high (102)
    res = validate_market_data(df, reference_date=REF)
    assert CAT_OHLC_INCONSISTENT in res.report.errors_by_category


@pytest.mark.unit
def test_invalid_ticker_rejected():
    df = good_df()
    df.loc[1, "ticker"] = "1$BAD"
    res = validate_market_data(df, reference_date=REF)
    assert CAT_INVALID_TICKER in res.report.errors_by_category


@pytest.mark.unit
def test_future_date_rejected():
    df = good_df()
    df.loc[2, "date"] = "2030-06-01"
    res = validate_market_data(df, reference_date=REF)
    assert CAT_FUTURE_DATE in res.report.errors_by_category


@pytest.mark.unit
def test_duplicate_row_is_warning_not_rejection():
    df = good_df()
    dup = df.iloc[[1]].copy()
    df = pd.concat([df, dup], ignore_index=True)
    res = validate_market_data(df, reference_date=REF)
    assert res.report.warnings_by_category.get(CAT_DUPLICATE_ROW) == 1
    assert res.report.rows_rejected == 0  # duplicates handled in cleaning, not rejected


@pytest.mark.unit
def test_abnormal_jump_warning():
    df = good_df()
    # Make the last close a ~100% jump from the prior close.
    df.loc[2, ["open", "high", "low", "close", "adjusted_close"]] = [150, 210, 149, 205, 205]
    res = validate_market_data(df, reference_date=REF, max_daily_move=0.5)
    assert res.report.warnings_by_category.get(CAT_ABNORMAL_JUMP, 0) >= 1
    assert res.report.rows_rejected == 0


@pytest.mark.unit
def test_missing_business_day_warning():
    df = good_df()
    df.loc[1, "date"] = "2024-01-09"  # skip several business days between 01-02 and 01-09
    res = validate_market_data(df, reference_date=REF)
    assert res.report.warnings_by_category.get(CAT_MISSING_DATE, 0) > 0


@pytest.mark.unit
def test_rejected_frame_has_reasons():
    df = good_df()
    df.loc[0, "close"] = np.nan
    res = validate_market_data(df, reference_date=REF)
    assert "reject_reasons" in res.rejected.columns
    assert res.rejected.iloc[0]["reject_reasons"]
