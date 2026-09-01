"""Unit tests for VaR (historical + parametric)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from app.core.exceptions import InsufficientHistoricalData
from app.risk.var import historical_var, parametric_var, var_to_dollars


@pytest.fixture()
def returns() -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0005, 0.02, size=500))


@pytest.mark.unit
def test_historical_var_equals_negative_quantile(returns):
    got = historical_var(returns, 0.95)
    expected = -float(np.quantile(returns.to_numpy(), 0.05))
    assert got == pytest.approx(expected)


@pytest.mark.unit
def test_historical_var_is_positive_loss(returns):
    assert historical_var(returns, 0.95) > 0


@pytest.mark.unit
def test_var_monotonic_in_confidence(returns):
    v90 = historical_var(returns, 0.90)
    v95 = historical_var(returns, 0.95)
    v99 = historical_var(returns, 0.99)
    assert v90 <= v95 <= v99  # higher confidence → larger loss threshold


@pytest.mark.unit
def test_parametric_var_formula():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0, 0.03, size=1000))
    sigma = r.std(ddof=1)
    got = parametric_var(r, 0.95, zero_mean=True)
    assert got == pytest.approx(norm.ppf(0.95) * sigma, rel=1e-9)


@pytest.mark.unit
def test_var_to_dollars():
    assert var_to_dollars(0.02, 100_000) == pytest.approx(2_000)


@pytest.mark.unit
def test_insufficient_data_raises():
    with pytest.raises(InsufficientHistoricalData):
        historical_var(pd.Series([0.01]), 0.95)


@pytest.mark.unit
def test_bad_confidence_raises():
    r = pd.Series(np.zeros(50) + 0.01)
    with pytest.raises(ValueError):
        historical_var(r, 1.5)
    with pytest.raises(ValueError):
        parametric_var(r, 0.0)
