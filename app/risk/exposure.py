"""Portfolio exposure and weights (pure functions).

Operates on a mapping of ticker → signed market value:

    gross = Σ_i |mv_i|          (total capital at risk, long + short)
    net   = Σ_i  mv_i           (directional exposure)
    w_i   = mv_i / gross        (signed weight; long-only weights sum to 1)
"""

from __future__ import annotations

from collections.abc import Mapping


def gross_exposure(market_values: Mapping[str, float]) -> float:
    """Sum of absolute market values."""
    return float(sum(abs(v) for v in market_values.values()))


def net_exposure(market_values: Mapping[str, float]) -> float:
    """Signed sum of market values."""
    return float(sum(market_values.values()))


def weights(market_values: Mapping[str, float]) -> dict[str, float]:
    """Signed weight of each holding relative to gross exposure.

    Uses gross as the denominator so a long/short book with ~zero net exposure still
    produces stable weights. Returns an empty dict when gross exposure is zero.
    """
    gross = gross_exposure(market_values)
    if gross == 0.0:
        return {}
    return {t: float(v) / gross for t, v in market_values.items()}


def net_weights(market_values: Mapping[str, float]) -> dict[str, float]:
    """Weights relative to *net* exposure (sum to 1 for long-only).

    These are the weights used for volatility risk decomposition, where the portfolio
    return is ``Σ w_i r_i`` with ``Σ w_i = 1``. Returns an empty dict if net is zero.
    """
    net = net_exposure(market_values)
    if net == 0.0:
        return {}
    return {t: float(v) / net for t, v in market_values.items()}
