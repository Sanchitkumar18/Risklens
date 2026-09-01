"""Unit tests for the explanation renderer (all intents) and the LLM path."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app.core.config import get_settings
from app.llm.nodes import classify as c
from app.llm.nodes.explain import make_explain, render_answer


@pytest.mark.unit
def test_render_var_explain():
    results = {
        "get_risk_metrics": {
            "confidence_level": 0.99, "observations": 1500, "portfolio_value": 184657.0,
            "volatility_annualized": 0.212,
            "var_historical": [
                {"confidence_level": 0.95, "var_fraction": 0.0202, "var_value": 3731.0},
                {"confidence_level": 0.99, "var_fraction": 0.0309, "var_value": 5715.0},
            ],
            "var_parametric": {"confidence_level": 0.99, "var_fraction": 0.0302, "var_value": 5585.0},
        }
    }
    ans = render_answer(c.VAR_EXPLAIN, "explain 99% var", results, 0.99)
    assert "99%" in ans and "$5,715" in ans and "not a maximum" in ans
    assert "not financial advice" in ans.lower()


@pytest.mark.unit
def test_render_contribution():
    results = {
        "get_risk_contributions": {
            "contributions": [
                {"ticker": "NVDA", "weight": 0.39, "percent": 0.55},
                {"ticker": "MSFT", "weight": 0.32, "percent": 0.23},
            ],
            "top_contributor": {"ticker": "NVDA", "weight": 0.39, "percent": 0.55},
        }
    }
    ans = render_answer(c.CONTRIBUTION, "which asset", results, None)
    assert "NVDA" in ans and "55.0%" in ans


@pytest.mark.unit
def test_render_drawdown():
    results = {"get_drawdown_analysis": {"max_drawdown": -0.373, "peak_date": "2022-10-05", "trough_date": "2024-01-09"}}
    ans = render_answer(c.DRAWDOWN, "drawdown?", results, None)
    assert "-37.3%" in ans and "2022-10-05" in ans


@pytest.mark.unit
def test_render_correlation_with_and_without_pairs():
    with_pairs = {"get_correlation_matrix": {"tickers": ["A", "B"], "matrix": {}, "high_correlation_pairs": [{"ticker_a": "A", "ticker_b": "B", "correlation": 0.85}]}}
    ans = render_answer(c.CORRELATION, "correlation", with_pairs, None)
    assert "0.85" in ans and "A–B" in ans

    without = {"get_correlation_matrix": {"tickers": ["A", "B"], "matrix": {}, "high_correlation_pairs": []}}
    ans2 = render_answer(c.CORRELATION, "correlation", without, None)
    assert "No asset pairs exceed" in ans2


@pytest.mark.unit
def test_render_exposure():
    results = {"get_asset_exposure": {"gross_exposure": 1000.0, "net_exposure": 1000.0, "largest_position": {"ticker": "NVDA", "weight": 0.4}}}
    ans = render_answer(c.EXPOSURE, "exposure", results, None)
    assert "$1,000" in ans and "NVDA" in ans


@pytest.mark.unit
def test_render_stress():
    results = {"run_stress_test": {"scenario": "market_crash", "description": "d", "total_loss": 100.0, "pct_loss": -0.2, "worst_assets": ["NVDA"], "legs": []}}
    ans = render_answer(c.STRESS, "crash", results, None)
    assert "market_crash" in ans and "-20.0%" in ans and "NVDA" in ans


@pytest.mark.unit
def test_render_anomalies_and_alerts_empty_and_present():
    assert "No unusual" in render_answer(c.ANOMALIES, "q", {"get_anomalies": {"count": 0, "anomalies": []}}, None)
    present = {"get_anomalies": {"count": 2, "anomalies": [{"ticker": "NVDA", "date": "2023-10-09", "type": "volume_spike", "score": 0.76}]}}
    assert "Detected 2" in render_answer(c.ANOMALIES, "q", present, None)

    assert "no active risk alerts" in render_answer(c.ALERTS, "q", {"get_alerts": {"count": 0, "alerts": []}}, None).lower()
    alerts = {"get_alerts": {"count": 1, "alerts": [{"severity": "HIGH", "message": "drawdown breach", "acknowledged": False}]}}
    assert "1 active alert" in render_answer(c.ALERTS, "q", alerts, None)


@pytest.mark.unit
def test_render_summary_and_empty_fallback():
    results = {
        "get_portfolio_summary": {"name": "P", "total_value": 100000.0, "num_positions": 3, "unrealized_pnl": 5000.0},
        "get_risk_metrics": {"confidence_level": 0.95, "volatility_annualized": 0.2, "var_historical": [{"confidence_level": 0.95, "var_fraction": 0.02, "var_value": 2000.0}]},
    }
    ans = render_answer(c.RISK_SUMMARY, "risky?", results, 0.95)
    assert "$100,000" in ans and "volatility" in ans.lower()

    empty = render_answer(c.RISK_SUMMARY, "risky?", {}, None)
    assert "don't have enough data" in empty


# ── LLM path ────────────────────────────────────────────────
class _StubModel:
    def invoke(self, messages):
        return AIMessage(content="Grounded: 95% VaR is $2,000.")


class _RaisingModel:
    def invoke(self, messages):
        raise RuntimeError("model down")


@pytest.mark.unit
def test_explain_uses_llm_when_model_present():
    node = make_explain(get_settings(), _StubModel())
    out = node({"intent": c.VAR_EXPLAIN, "question": "q", "tool_results": {"x": 1}, "confidence": 0.95})
    assert out["grounded"] is False
    assert "VaR" in out["draft_answer"]


@pytest.mark.unit
def test_explain_falls_back_when_model_errors():
    results = {"get_drawdown_analysis": {"max_drawdown": -0.2, "peak_date": "2022-01-01", "trough_date": "2022-06-01"}}
    node = make_explain(get_settings(), _RaisingModel())
    out = node({"intent": c.DRAWDOWN, "question": "q", "tool_results": results, "confidence": None})
    assert out["grounded"] is True  # fell back to deterministic renderer
    assert "-20.0%" in out["draft_answer"]
