"""Unit tests for drawdown."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.risk.drawdown import drawdown_series, max_drawdown, max_drawdown_detail


@pytest.fixture()
def values() -> pd.Series:
    idx = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]
    )
    return pd.Series([100.0, 120.0, 90.0, 110.0, 80.0, 130.0], index=idx)


@pytest.mark.unit
def test_max_drawdown_value(values):
    # Peak 120 → trough 80 = -1/3.
    assert max_drawdown(values) == pytest.approx(-1 / 3)


@pytest.mark.unit
def test_drawdown_series_never_positive(values):
    dd = drawdown_series(values)
    assert (dd <= 1e-12).all()


@pytest.mark.unit
def test_max_drawdown_detail_dates(values):
    res = max_drawdown_detail(values)
    assert res.max_drawdown == pytest.approx(-1 / 3)
    assert res.peak_date == date(2024, 1, 3)     # value 120
    assert res.trough_date == date(2024, 1, 8)   # value 80


@pytest.mark.unit
def test_monotonic_increasing_has_zero_drawdown():
    v = pd.Series([100.0, 101.0, 102.0, 103.0])
    assert max_drawdown(v) == pytest.approx(0.0)
