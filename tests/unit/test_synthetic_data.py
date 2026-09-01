"""Unit tests for the synthetic market-data generator.

These assert the *statistical character* the risk engine relies on: reproducibility,
valid OHLC bars, cross-asset correlation, a volatility ordering (SPY calmer than NVDA),
volatility clustering, and the presence of injected tail events.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.pipelines.synthetic_data import (
    DEFAULT_ASSETS,
    generate_market_data,
)

START, END = "2020-01-01", "2023-12-31"


@pytest.fixture(scope="module")
def data():
    return generate_market_data(start=START, end=END, seed=42)


def _returns(df, ticker):
    s = df[df["ticker"] == ticker].sort_values("date")["adjusted_close"].astype(float)
    return s.pct_change().dropna().to_numpy()


@pytest.mark.unit
def test_reproducible_with_same_seed():
    a = generate_market_data(start=START, end=END, seed=7)
    b = generate_market_data(start=START, end=END, seed=7)
    assert a.equals(b)


@pytest.mark.unit
def test_different_seed_differs():
    a = generate_market_data(start=START, end=END, seed=1)
    b = generate_market_data(start=START, end=END, seed=2)
    assert not a["close"].equals(b["close"])


@pytest.mark.unit
def test_shape_and_columns(data):
    tickers = sorted(a.ticker for a in DEFAULT_ASSETS)
    assert sorted(data["ticker"].unique()) == tickers
    assert list(data.columns) == [
        "date", "ticker", "open", "high", "low", "close", "adjusted_close", "volume",
    ]
    # equal-length series per ticker, no missing values
    counts = data.groupby("ticker").size().unique()
    assert len(counts) == 1
    assert not data.isna().any().any()


@pytest.mark.unit
def test_ohlc_consistency(data):
    assert (data["high"] >= data["low"]).all()
    assert (data["high"] >= data["open"]).all()
    assert (data["high"] >= data["close"]).all()
    assert (data["low"] <= data["open"]).all()
    assert (data["low"] <= data["close"]).all()
    for col in ["open", "high", "low", "close", "adjusted_close"]:
        assert (data[col] > 0).all()
    assert (data["volume"] > 0).all()


@pytest.mark.unit
def test_volatility_ordering(data):
    """SPY (broad market) should be less volatile than NVDA (high-beta tech)."""
    spy_vol = _returns(data, "SPY").std() * np.sqrt(252)
    nvda_vol = _returns(data, "NVDA").std() * np.sqrt(252)
    assert spy_vol < nvda_vol
    assert 0.05 < spy_vol < 0.45  # plausible annualized range


@pytest.mark.unit
def test_cross_asset_correlation(data):
    """Assets sharing factors are positively correlated (tech pair + market)."""
    aapl, msft, spy = _returns(data, "AAPL"), _returns(data, "MSFT"), _returns(data, "SPY")
    assert np.corrcoef(aapl, msft)[0, 1] > 0.3
    assert np.corrcoef(aapl, spy)[0, 1] > 0.3


@pytest.mark.unit
def test_volatility_clustering(data):
    """Squared returns are positively autocorrelated (GARCH clustering)."""
    r = _returns(data, "SPY")
    sq = r**2
    lag1_autocorr = np.corrcoef(sq[:-1], sq[1:])[0, 1]
    assert lag1_autocorr > 0.0


@pytest.mark.unit
def test_tail_events_present(data):
    """Injected crashes/jumps produce at least one large single-day move."""
    min_ret = min(_returns(data, a.ticker).min() for a in DEFAULT_ASSETS)
    assert min_ret < -0.07
