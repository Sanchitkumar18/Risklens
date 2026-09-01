"""Asset-level contribution to portfolio volatility (Euler / covariance method).

Portfolio variance is ``σ_p² = wᵀ Σ w`` where ``w`` are net weights (summing to 1)
and ``Σ`` is the return covariance matrix. Because volatility is homogeneous of
degree 1 in the weights, Euler's theorem gives an **exact additive decomposition**:

    marginal contribution   MCR_i = (Σ w)_i / σ_p
    component contribution   CCR_i = w_i · MCR_i          with   Σ_i CCR_i = σ_p
    percent contribution     PCR_i = CCR_i / σ_p          with   Σ_i PCR_i = 1

This is *the* correct way to answer "which asset contributes most to portfolio risk?"
— it accounts for correlations, not just each asset's standalone volatility. A highly
volatile but diversifying asset can contribute *less* risk than its weight suggests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.core.exceptions import RiskCalculationError


@dataclass
class AssetContribution:
    ticker: str
    weight: float
    marginal: float   # marginal contribution to risk (per unit weight)
    component: float  # absolute risk contribution (same units as σ_p)
    percent: float    # share of total portfolio volatility (0..1)


@dataclass
class ContributionResult:
    portfolio_volatility: float          # σ_p in the same period units as the returns
    contributions: list[AssetContribution]


def risk_contributions(
    returns: pd.DataFrame, weights: Mapping[str, float]
) -> ContributionResult:
    """Decompose portfolio volatility into per-asset contributions.

    Args:
        returns: wide ``date × ticker`` returns matrix.
        weights: net weight per ticker (should sum to ~1); must cover the columns used.

    Returns:
        A :class:`ContributionResult` whose component contributions sum to the
        portfolio volatility (Euler identity).
    """
    tickers = [t for t in returns.columns if t in weights]
    if not tickers:
        raise RiskCalculationError("No overlap between returns columns and weights.")

    w = np.array([float(weights[t]) for t in tickers], dtype=float)
    cov = returns[tickers].cov().to_numpy(dtype=float)  # sample covariance (ddof=1)

    variance = float(w @ cov @ w)
    sigma_p = float(np.sqrt(variance)) if variance > 0 else 0.0

    if sigma_p == 0.0:
        # Degenerate (e.g. constant returns): no risk to attribute.
        contribs = [
            AssetContribution(ticker=t, weight=float(wi), marginal=0.0, component=0.0, percent=0.0)
            for t, wi in zip(tickers, w, strict=True)
        ]
        return ContributionResult(portfolio_volatility=0.0, contributions=contribs)

    marginal = (cov @ w) / sigma_p          # MCR_i
    component = w * marginal                # CCR_i,  Σ = σ_p
    percent = component / sigma_p           # PCR_i,  Σ = 1

    contributions = [
        AssetContribution(
            ticker=t,
            weight=float(w[i]),
            marginal=float(marginal[i]),
            component=float(component[i]),
            percent=float(percent[i]),
        )
        for i, t in enumerate(tickers)
    ]
    contributions.sort(key=lambda c: c.component, reverse=True)
    return ContributionResult(portfolio_volatility=sigma_p, contributions=contributions)
