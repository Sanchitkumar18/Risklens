"""Unit tests for volatility."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import InsufficientHistoricalData
from app.risk.volatility import annualized_volatility, daily_volatility


@pytest.mark.unit
def test_daily_volatility_known_value():
    r = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0])
    expected = r.std(ddof=1)  # unbiased sample std
    assert daily_volatility(r) == pytest.approx(expected)
    assert daily_volatility(r) == pytest.approx(0.0158113883)


@pytest.mark.unit
def test_annualized_scales_by_sqrt_252():
    r = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0])
    assert annualized_volatility(r) == pytest.approx(daily_volatility(r) * math.sqrt(252))


@pytest.mark.unit
def test_constant_returns_zero_volatility():
    r = pd.Series(np.full(30, 0.005))
    assert daily_volatility(r) == pytest.approx(0.0)
    assert annualized_volatility(r) == pytest.approx(0.0)


@pytest.mark.unit
def test_insufficient_observations_raises():
    with pytest.raises(InsufficientHistoricalData):
        daily_volatility(pd.Series([0.01]))
