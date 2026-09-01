"""Unit tests for the transformation / feature-engineering stage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.pipelines.synthetic_data import generate_market_data
from app.pipelines.transformation import (
    add_returns,
    enrich,
    to_returns_matrix,
)


def _series_df(prices: list[float], ticker: str = "AAPL") -> pd.DataFrame:
    n = len(prices)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=n),
            "ticker": [ticker] * n,
            "open": prices, "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices], "close": prices,
            "adjusted_close": prices, "volume": [1000] * n,
        }
    )


@pytest.mark.unit
def test_add_returns_values():
    df = _series_df([100.0, 110.0, 121.0])
    out = add_returns(df)
    assert np.isnan(out["simple_return"].iloc[0])
    assert out["simple_return"].iloc[1] == pytest.approx(0.10)
    assert out["simple_return"].iloc[2] == pytest.approx(0.10)
    assert out["log_return"].iloc[1] == pytest.approx(np.log(1.10))


@pytest.mark.unit
def test_returns_are_per_ticker():
    a = _series_df([100.0, 110.0], "AAPL")
    b = _series_df([200.0, 180.0], "MSFT")
    out = add_returns(pd.concat([a, b], ignore_index=True))
    # First row of each ticker must be NaN (no cross-ticker leakage).
    # groupby().head(1) keeps NaN (unlike .first(), which skips it).
    first_rows = out.groupby("ticker", sort=False).head(1)
    assert first_rows["simple_return"].isna().all()


@pytest.mark.unit
def test_enrich_adds_expected_columns():
    df = generate_market_data(start="2022-01-03", end="2022-12-31", seed=5)
    out = enrich(df)
    for col in [
        "simple_return", "log_return", "rolling_vol_21",
        "ma_20", "ma_50", "dist_from_ma_20", "volume_change",
    ]:
        assert col in out.columns
    # With ~250 rows/ticker, rolling features must have non-null values.
    assert out["rolling_vol_21"].notna().any()
    assert out["ma_50"].notna().any()


@pytest.mark.unit
def test_returns_matrix_shape():
    df = generate_market_data(start="2022-01-03", end="2022-03-31", seed=5)
    mat = to_returns_matrix(df)
    assert set(mat.columns) == set(df["ticker"].unique())
    assert mat.index.is_monotonic_increasing
