"""Value at Risk (VaR).

**Sign convention (used everywhere):** VaR is returned as a **positive loss**. A 95%
1-day VaR of 0.021 means "a one-day loss of ~2.1% or worse occurred about 5% of the
time in the sample." Multiply by portfolio value for a dollar figure. VaR is *not* a
maximum-loss guarantee — losses beyond VaR happen with probability ≈ (1 − confidence).

Two methodologies are implemented:

* **Historical simulation** (primary) — the empirical ``(1 − c)`` quantile of realized
  returns. No distributional assumption; captures fat tails and skew present in the
  data. Needs enough observations to estimate a tail quantile.
* **Parametric (variance–covariance)** — assumes returns are Normal:
  ``VaR = −(μ + z_{1−c}·σ)``. Provided for comparison/teaching; understates tail risk
  when returns are fat-tailed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from app.core.exceptions import InsufficientHistoricalData

_MIN_OBS_HISTORICAL = 2
_MIN_OBS_PARAMETRIC = 2


def _clean(returns: pd.Series) -> pd.Series:
    return pd.Series(returns, dtype="float64").dropna()


def _check_confidence(confidence_level: float) -> None:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")


def historical_var(
    returns: pd.Series, confidence_level: float = 0.95, min_observations: int = _MIN_OBS_HISTORICAL
) -> float:
    """Historical-simulation VaR as a positive loss fraction.

    ``VaR_c = −Quantile_{1−c}(returns)``. With ``confidence_level=0.95`` this is the
    negative of the empirical 5th percentile of returns.
    """
    _check_confidence(confidence_level)
    r = _clean(returns)
    if len(r) < min_observations:
        raise InsufficientHistoricalData(
            "Not enough observations for historical VaR.",
            details={"observations": len(r), "required": min_observations},
        )
    quantile = float(np.quantile(r.to_numpy(), 1.0 - confidence_level))
    return -quantile


def parametric_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
    zero_mean: bool = False,
    min_observations: int = _MIN_OBS_PARAMETRIC,
) -> float:
    """Parametric (Normal) VaR as a positive loss fraction.

    ``VaR_c = −(μ + z_{1−c}·σ)`` with ``z_{1−c} = Φ⁻¹(1−c)`` (a negative number).
    Set ``zero_mean=True`` to drop the drift term (common for short horizons).
    """
    _check_confidence(confidence_level)
    r = _clean(returns)
    if len(r) < min_observations:
        raise InsufficientHistoricalData(
            "Not enough observations for parametric VaR.",
            details={"observations": len(r), "required": min_observations},
        )
    mu = 0.0 if zero_mean else float(r.mean())
    sigma = float(r.std(ddof=1))
    z = float(norm.ppf(1.0 - confidence_level))
    return -(mu + z * sigma)


def var_to_dollars(var_fraction: float, portfolio_value: float) -> float:
    """Convert a fractional VaR into a dollar loss for a given portfolio value."""
    return var_fraction * abs(portfolio_value)
