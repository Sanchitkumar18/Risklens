"""Unit tests for risk contribution (Euler decomposition)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analytics.risk_contribution import risk_contributions


@pytest.fixture()
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    # Two correlated assets.
    common = rng.normal(0, 0.01, 300)
    a = common + rng.normal(0, 0.005, 300)
    b = 0.5 * common + rng.normal(0, 0.02, 300)
    return pd.DataFrame({"A": a, "B": b})


@pytest.mark.unit
def test_components_sum_to_portfolio_volatility(returns):
    weights = {"A": 0.6, "B": 0.4}
    res = risk_contributions(returns, weights)

    # Independent check of σ_p = sqrt(wᵀ Σ w).
    w = np.array([0.6, 0.4])
    cov = returns[["A", "B"]].cov().to_numpy()
    sigma_p = float(np.sqrt(w @ cov @ w))

    assert res.portfolio_volatility == pytest.approx(sigma_p)
    total_component = sum(c.component for c in res.contributions)
    assert total_component == pytest.approx(sigma_p)  # Euler identity


@pytest.mark.unit
def test_percent_contributions_sum_to_one(returns):
    res = risk_contributions(returns, {"A": 0.6, "B": 0.4})
    assert sum(c.percent for c in res.contributions) == pytest.approx(1.0)


@pytest.mark.unit
def test_single_asset_is_full_contribution():
    r = pd.DataFrame({"A": np.random.default_rng(0).normal(0, 0.01, 100)})
    res = risk_contributions(r, {"A": 1.0})
    assert res.contributions[0].percent == pytest.approx(1.0)


@pytest.mark.unit
def test_constant_returns_zero_risk():
    r = pd.DataFrame({"A": np.full(50, 0.01), "B": np.full(50, 0.02)})
    res = risk_contributions(r, {"A": 0.5, "B": 0.5})
    assert res.portfolio_volatility == pytest.approx(0.0)
    assert all(c.percent == 0.0 for c in res.contributions)
