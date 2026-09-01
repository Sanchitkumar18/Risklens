"""Unit tests for the pure stress-testing engine."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.risk.stress_testing import (
    MARKET_CRASH,
    TECH_SELLOFF,
    VOLATILITY_SHOCK,
    ScenarioSpec,
    StressHolding,
    apply_stress,
    builtin_scenarios,
    default_classifier,
    resolve_shocks,
)


def _holdings() -> list[StressHolding]:
    return [
        StressHolding("AAPL", Decimal("100"), Decimal("100")),  # $10,000
        StressHolding("SPY", Decimal("10"), Decimal("400")),    # $4,000
    ]


@pytest.mark.unit
def test_classifier():
    assert default_classifier("NVDA") == "technology"
    assert default_classifier("SPY") == "broad"
    assert default_classifier("XOM") == "equity"


@pytest.mark.unit
def test_apply_stress_pnl_math():
    holdings = _holdings()
    out = apply_stress(holdings, {"AAPL": -0.10, "SPY": -0.20})
    # AAPL: 10000 * -0.10 = -1000 ; SPY: 4000 * -0.20 = -800
    assert out.total_pnl == Decimal("-1800.00")
    assert out.total_loss == Decimal("1800.00")
    assert out.portfolio_value_before == Decimal("14000.00")
    assert out.pct_loss == pytest.approx(-1800 / 14000)


@pytest.mark.unit
def test_worst_assets_ordering():
    out = apply_stress(_holdings(), {"AAPL": -0.10, "SPY": -0.20})
    # AAPL loses more in absolute terms ($1000 > $800) → listed first.
    assert out.worst_assets[0] == "AAPL"
    assert out.legs[0].ticker == "AAPL"


@pytest.mark.unit
def test_resolve_precedence_ticker_over_class_over_default():
    spec = ScenarioSpec(
        name="x", description="",
        ticker_shocks={"AAPL": -0.5},
        class_shocks={"technology": -0.25},
        default_shock=-0.1,
    )
    shocks = resolve_shocks(spec, ["AAPL", "NVDA", "XOM"])
    assert shocks["AAPL"] == -0.5    # ticker-specific wins
    assert shocks["NVDA"] == -0.25   # class (technology)
    assert shocks["XOM"] == -0.1     # default


@pytest.mark.unit
def test_market_crash_hits_everything_20pct():
    spec = builtin_scenarios()[MARKET_CRASH]
    out = apply_stress(_holdings(), resolve_shocks(spec, ["AAPL", "SPY"]),
                       scenario_name=spec.name)
    assert out.pct_loss == pytest.approx(-0.20)


@pytest.mark.unit
def test_tech_selloff_differentiates():
    spec = builtin_scenarios()[TECH_SELLOFF]
    shocks = resolve_shocks(spec, ["AAPL", "SPY"])
    assert shocks["AAPL"] == -0.25   # technology
    assert shocks["SPY"] == -0.10    # default (broad not in class_shocks)


@pytest.mark.unit
def test_volatility_shock_uses_sigma():
    spec = builtin_scenarios()[VOLATILITY_SHOCK]  # vol_multiple = 3
    shocks = resolve_shocks(spec, ["AAPL", "SPY"], sigma_map={"AAPL": 0.02, "SPY": 0.01})
    assert shocks["AAPL"] == pytest.approx(-0.06)  # -3 * 0.02
    assert shocks["SPY"] == pytest.approx(-0.03)   # -3 * 0.01


@pytest.mark.unit
def test_positive_shock_is_a_gain():
    out = apply_stress(_holdings(), {"AAPL": 0.10, "SPY": 0.10})
    assert out.total_pnl > 0
    assert out.total_loss == Decimal("0.00")
