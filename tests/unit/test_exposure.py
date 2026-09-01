"""Unit tests for exposure and weights."""

from __future__ import annotations

import pytest

from app.risk.exposure import gross_exposure, net_exposure, net_weights, weights


@pytest.mark.unit
def test_long_only_exposure_and_weights():
    mv = {"AAPL": 1000.0, "MSFT": 3000.0}
    assert gross_exposure(mv) == pytest.approx(4000.0)
    assert net_exposure(mv) == pytest.approx(4000.0)
    w = weights(mv)
    assert w["AAPL"] == pytest.approx(0.25)
    assert w["MSFT"] == pytest.approx(0.75)
    assert sum(w.values()) == pytest.approx(1.0)


@pytest.mark.unit
def test_long_short_gross_vs_net():
    mv = {"AAPL": 1000.0, "MSFT": -1000.0}
    assert gross_exposure(mv) == pytest.approx(2000.0)
    assert net_exposure(mv) == pytest.approx(0.0)
    w = weights(mv)
    assert w["AAPL"] == pytest.approx(0.5)
    assert w["MSFT"] == pytest.approx(-0.5)


@pytest.mark.unit
def test_net_weights_sum_to_one_long_only():
    mv = {"AAPL": 1000.0, "MSFT": 3000.0}
    nw = net_weights(mv)
    assert sum(nw.values()) == pytest.approx(1.0)


@pytest.mark.unit
def test_zero_gross_returns_empty_weights():
    assert weights({}) == {}
    assert net_weights({"A": 1000.0, "B": -1000.0}) == {}
