"""Unit tests for the pure alert-rule evaluation."""

from __future__ import annotations

from datetime import date

import pytest

from app.analytics.alert_rules import build_alert_candidates, grade_severity
from app.core.enums import AlertSeverity, AlertType
from app.schemas.alert import AlertThresholds
from app.schemas.risk import DrawdownResultSchema, RiskReport, VarResult

AS_OF = date(2024, 12, 31)


def _report(**overrides) -> RiskReport:
    base = dict(
        portfolio_id=1, name="P", as_of_date=AS_OF, observations=500, confidence_level=0.95,
        portfolio_value=100_000.0, gross_exposure=100_000.0, net_exposure=100_000.0,
        volatility_daily=0.009, volatility_annualized=0.15,
        var_historical=[VarResult(confidence_level=0.95, method="historical", var_fraction=0.010, var_value=1000.0)],
        var_parametric=None,
        drawdown=DrawdownResultSchema(max_drawdown=-0.08, peak_date=None, trough_date=None),
        weights={"AAPL": 0.30, "MSFT": 0.30, "GOOGL": 0.30},
        correlation_matrix={}, high_correlation_pairs=[], risk_contributions=[],
    )
    base.update(overrides)
    return RiskReport(**base)


@pytest.mark.unit
def test_grade_severity_bands():
    assert grade_severity(0.32, 0.20) == AlertSeverity.CRITICAL.value   # ratio 1.6
    assert grade_severity(0.26, 0.20) == AlertSeverity.HIGH.value       # ratio 1.3
    assert grade_severity(0.21, 0.20) == AlertSeverity.MEDIUM.value     # ratio 1.05
    assert grade_severity(0.19, 0.20) == AlertSeverity.LOW.value        # ratio 0.95
    assert grade_severity(0.10, 0.20) is None                           # well within


@pytest.mark.unit
def test_no_breaches_within_limits():
    report = _report()  # vol 0.18<0.20, var 0.015<0.02, dd 0.10<0.25, weights<0.35
    candidates = build_alert_candidates(
        portfolio_id=1, as_of=AS_OF, report=report, thresholds=AlertThresholds()
    )
    assert candidates == []


@pytest.mark.unit
def test_var_breach_detected():
    report = _report(
        var_historical=[VarResult(confidence_level=0.95, method="historical", var_fraction=0.03, var_value=3000.0)]
    )
    cands = build_alert_candidates(portfolio_id=1, as_of=AS_OF, report=report, thresholds=AlertThresholds())
    types = {c.alert_type for c in cands}
    assert AlertType.VAR_BREACH.value in types


@pytest.mark.unit
def test_concentration_and_drawdown_breach():
    report = _report(
        weights={"NVDA": 0.5, "AAPL": 0.5},
        drawdown=DrawdownResultSchema(max_drawdown=-0.40, peak_date=None, trough_date=None),
    )
    cands = build_alert_candidates(portfolio_id=1, as_of=AS_OF, report=report, thresholds=AlertThresholds())
    types = {c.alert_type for c in cands}
    assert AlertType.CONCENTRATION.value in types
    assert AlertType.DRAWDOWN_BREACH.value in types
    # Drawdown 0.40 / 0.25 = 1.6 → CRITICAL.
    dd = next(c for c in cands if c.alert_type == AlertType.DRAWDOWN_BREACH.value)
    assert dd.severity == AlertSeverity.CRITICAL.value


@pytest.mark.unit
def test_dedup_keys_are_stable_and_distinct():
    report = _report(weights={"NVDA": 0.5, "AAPL": 0.5})
    cands = build_alert_candidates(portfolio_id=1, as_of=AS_OF, report=report, thresholds=AlertThresholds())
    keys = [c.dedup_key for c in cands]
    assert len(keys) == len(set(keys))  # unique
    assert all(str(AS_OF) in k and k.startswith("1:") for k in keys)
