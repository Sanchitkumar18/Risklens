"""Unit tests for assistant nodes: classify, plan, and the grounding validator."""

from __future__ import annotations

import pytest

from app.llm.nodes.classify import (
    ALERTS,
    ANOMALIES,
    CONTRIBUTION,
    DRAWDOWN,
    STRESS,
    VAR_EXPLAIN,
    classify_intent,
)
from app.llm.nodes.plan import choose_scenario
from app.llm.nodes.validate import find_ungrounded_numbers


@pytest.mark.unit
@pytest.mark.parametrize(
    "question,expected",
    [
        ("What are my current risk alerts?", ALERTS),
        ("Which assets have unusual movements?", ANOMALIES),
        ("Which asset contributes most to my portfolio risk?", CONTRIBUTION),
        ("What caused the largest drawdown?", DRAWDOWN),
        ("What happens if technology stocks fall 25%?", STRESS),
        ("Explain my 99% VaR in simple terms.", VAR_EXPLAIN),
    ],
)
def test_classify_intent(question, expected):
    assert classify_intent(question) == expected


@pytest.mark.unit
def test_choose_scenario():
    assert choose_scenario("what if technology stocks fall") == "tech_selloff"
    assert choose_scenario("a severe crash") == "severe_crash"
    assert choose_scenario("a volatility spike") == "volatility_shock"
    assert choose_scenario("a market crash") == "market_crash"


@pytest.mark.unit
def test_grounding_passes_for_tool_numbers():
    results = {
        "get_risk_metrics": {
            "volatility_annualized": 0.212,
            "var_historical": [{"confidence_level": 0.95, "var_value": 3731.0, "var_fraction": 0.0202}],
        }
    }
    text = "Annualized volatility is 21.2% and the 95% VaR is about $3,731."
    assert find_ungrounded_numbers(text, results) == []


@pytest.mark.unit
def test_grounding_flags_fabricated_number():
    results = {"get_risk_metrics": {"var_historical": [{"var_value": 3731.0}]}}
    text = "The VaR is actually $9,999."
    flagged = find_ungrounded_numbers(text, results)
    assert 9999.0 in flagged


@pytest.mark.unit
def test_grounding_ignores_iso_dates():
    results = {"get_drawdown_analysis": {"max_drawdown": -0.373}}
    text = "Max drawdown -37.3% with a trough on 2022-10-05."
    assert find_ungrounded_numbers(text, results) == []
