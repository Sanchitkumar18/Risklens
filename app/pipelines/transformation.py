"""Feature-engineering / transformation stage.

Derives the analytical columns that downstream layers consume — daily and log
returns, rolling volatility, moving averages, distance-from-average, and volume
change. These features are **computed on read** from the stored OHLCV bars rather
than persisted, so there is a single source of truth (raw bars) and no risk of stale
derived data; the cost is a cheap, vectorized recomputation per query.

All operations are grouped by ticker and vectorized (no Python row loops).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_VOL_WINDOW = 21   # ~1 trading month
DEFAULT_MA_WINDOWS = (20, 50)


def add_returns(df: pd.DataFrame, price_col: str = "adjusted_close") -> pd.DataFrame:
    """Add ``simple_return`` and ``log_return`` per ticker.

    Simple return ``P_t/P_{t-1} - 1`` is used for portfolio aggregation; log return
    ``ln(P_t/P_{t-1})`` is convenient for time aggregation and is roughly additive.
    """
    out = df.sort_values(["ticker", "date"], kind="stable").copy()
    grouped = out.groupby("ticker", sort=False)[price_col]
    out["simple_return"] = grouped.pct_change(fill_method=None)
    out["log_return"] = np.log(out[price_col] / grouped.shift(1))
    return out


def add_rolling_features(
    df: pd.DataFrame,
    vol_window: int = DEFAULT_VOL_WINDOW,
    ma_windows: tuple[int, ...] = DEFAULT_MA_WINDOWS,
    price_col: str = "adjusted_close",
) -> pd.DataFrame:
    """Add rolling volatility, moving averages, distance-from-MA, and volume change.

    Requires ``simple_return`` (call :func:`add_returns` first, or use :func:`enrich`).
    """
    out = df.sort_values(["ticker", "date"], kind="stable").copy()
    if "simple_return" not in out.columns:
        out = add_returns(out, price_col=price_col)

    g = out.groupby("ticker", sort=False)

    # Rolling volatility of simple returns (sample std over the window).
    out[f"rolling_vol_{vol_window}"] = g["simple_return"].transform(
        lambda s: s.rolling(vol_window, min_periods=max(2, vol_window // 2)).std()
    )

    for w in ma_windows:
        col = f"ma_{w}"
        out[col] = g[price_col].transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())

    # Distance of price from the shortest moving average (mean-reversion signal).
    short_ma = f"ma_{min(ma_windows)}"
    out[f"dist_from_{short_ma}"] = (out[price_col] - out[short_ma]) / out[short_ma]

    # Day-over-day volume change.
    out["volume_change"] = g["volume"].pct_change(fill_method=None)

    return out


def enrich(
    df: pd.DataFrame,
    vol_window: int = DEFAULT_VOL_WINDOW,
    ma_windows: tuple[int, ...] = DEFAULT_MA_WINDOWS,
    price_col: str = "adjusted_close",
) -> pd.DataFrame:
    """Full transformation: returns + rolling features, sorted by (ticker, date)."""
    out = add_returns(df, price_col=price_col)
    out = add_rolling_features(out, vol_window=vol_window, ma_windows=ma_windows, price_col=price_col)
    return out


def to_price_matrix(df: pd.DataFrame, price_col: str = "adjusted_close") -> pd.DataFrame:
    """Pivot long OHLCV into a wide date × ticker price matrix (analytics input)."""
    matrix = df.pivot_table(index="date", columns="ticker", values=price_col, aggfunc="last")
    return matrix.sort_index()


def to_returns_matrix(df: pd.DataFrame, price_col: str = "adjusted_close") -> pd.DataFrame:
    """Wide date × ticker simple-returns matrix (correlation / risk input)."""
    prices = to_price_matrix(df, price_col=price_col)
    return prices.pct_change(fill_method=None).dropna(how="all")
