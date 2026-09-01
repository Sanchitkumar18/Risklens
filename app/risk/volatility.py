"""Volatility.

Daily volatility is the sample standard deviation of daily returns; annualized
volatility scales it by ``√(periods_per_year)`` (the square-root-of-time rule, valid
when returns are serially uncorrelated):

    σ_daily      = std(returns, ddof=1)
    σ_annualized = σ_daily · √252
"""

from __future__ import annotations

import math

import pandas as pd

from app.core.exceptions import InsufficientHistoricalData


def daily_volatility(returns: pd.Series, ddof: int = 1, min_observations: int = 2) -> float:
    """Sample standard deviation of daily returns (``ddof=1`` → unbiased)."""
    r = pd.Series(returns, dtype="float64").dropna()
    if len(r) < min_observations:
        raise InsufficientHistoricalData(
            "Not enough observations for volatility.",
            details={"observations": len(r), "required": min_observations},
        )
    return float(r.std(ddof=ddof))


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = 252, ddof: int = 1, min_observations: int = 2
) -> float:
    """Annualized volatility via the square-root-of-time rule."""
    return daily_volatility(returns, ddof=ddof, min_observations=min_observations) * math.sqrt(
        periods_per_year
    )
