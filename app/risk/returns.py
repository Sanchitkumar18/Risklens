"""Portfolio return-series construction (pure functions).

Implements the **current-holdings / historical-prices** convention agreed in the
design: the portfolio's historical value is reconstructed by applying *today's* share
quantities to each asset's historical adjusted-close, then differenced into a return
series. This answers "how risky is the book I hold now?" and makes returns additive
across assets within a period.

    V_t   = Σ_i  quantity_i · adj_close_{i,t}
    r^P_t = V_t / V_{t-1} − 1
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from app.core.exceptions import InsufficientHistoricalData, MarketDataNotFound


def build_portfolio_values(
    prices: pd.DataFrame, quantities: Mapping[str, float]
) -> pd.Series:
    """Reconstruct the portfolio value series from a price matrix and holdings.

    Args:
        prices: wide ``date × ticker`` matrix of adjusted-close prices.
        quantities: signed share count per ticker (held constant through history).

    Returns:
        A value series indexed by the dates on which *all* held tickers have prices
        (the common window), so the return series has no gaps.
    """
    tickers = list(quantities.keys())
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise MarketDataNotFound(
            "Price history missing for held tickers.", details={"tickers": sorted(missing)}
        )

    sub = prices[tickers].dropna(how="any")
    if sub.empty:
        raise InsufficientHistoricalData(
            "No overlapping price history across held tickers.",
            details={"tickers": tickers},
        )

    q = np.array([float(quantities[t]) for t in tickers], dtype=float)
    values = sub.to_numpy(dtype=float) @ q
    return pd.Series(values, index=sub.index, name="portfolio_value")


def portfolio_returns(values: pd.Series) -> pd.Series:
    """Simple daily returns of a value series (NaNs dropped)."""
    return values.pct_change(fill_method=None).dropna()


def asset_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns matrix from a wide price matrix (aligned, NaNs dropped)."""
    return prices.pct_change(fill_method=None).dropna(how="any")
