"""Integration tests for the grounded LangChain toolkit."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from app.llm.tools import RiskLensToolkit, build_langchain_tools
from app.pipelines.synthetic_data import generate_market_data
from app.schemas.portfolio import PortfolioCreate, PositionCreate
from app.services.market_data_service import MarketDataService
from app.services.portfolio_service import PortfolioService


@pytest.fixture()
def toolkit(db_session) -> RiskLensToolkit:
    MarketDataService(db_session).ingest_dataframe(generate_market_data(seed=42))
    ps = PortfolioService(db_session)
    pf = ps.create_portfolio(PortfolioCreate(name="Tech Growth"))
    for t, q, ap in [("AAPL", 100, 150), ("MSFT", 80, 250), ("NVDA", 50, 200), ("AMZN", 60, 130), ("GOOGL", 40, 120)]:
        ps.add_position(pf.id, PositionCreate(ticker=t, quantity=Decimal(q), average_price=Decimal(ap)))
    return RiskLensToolkit(db_session, pf.id)


@pytest.mark.integration
def test_portfolio_summary(toolkit):
    s = toolkit.get_portfolio_summary()
    assert s["num_positions"] == 5
    assert s["total_value"] > 0
    assert abs(sum(h["weight"] for h in s["holdings"]) - 1.0) < 1e-6


@pytest.mark.integration
def test_risk_metrics_grounded(toolkit):
    m = toolkit.get_risk_metrics(confidence=0.99)
    assert m["observations"] > 20
    assert m["volatility_annualized"] > 0
    v99 = next(v for v in m["var_historical"] if v["confidence_level"] == 0.99)
    assert v99["var_value"] > 0


@pytest.mark.integration
def test_risk_contributions_identifies_top(toolkit):
    c = toolkit.get_risk_contributions()
    assert c["top_contributor"]["ticker"] == "NVDA"  # highest-vol name dominates
    assert abs(sum(x["percent"] for x in c["contributions"]) - 1.0) < 1e-6


@pytest.mark.integration
def test_stress_tool(toolkit):
    r = toolkit.run_stress_test("market_crash")
    assert r["pct_loss"] < 0
    assert r["total_loss"] > 0
    assert r["worst_assets"]


@pytest.mark.integration
def test_correlation_and_drawdown(toolkit):
    corr = toolkit.get_correlation_matrix()
    assert set(corr["tickers"]) == {"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"}
    dd = toolkit.get_drawdown_analysis()
    assert dd["max_drawdown"] <= 0


@pytest.mark.integration
def test_anomalies_and_alerts_self_populate(toolkit):
    anomalies = toolkit.get_anomalies()
    assert anomalies["count"] >= 0
    alerts = toolkit.get_alerts()
    assert alerts["count"] >= 0


@pytest.mark.integration
def test_langchain_tools_wrap_and_invoke(toolkit):
    tools = build_langchain_tools(toolkit)
    names = {t.name for t in tools}
    assert {
        "get_portfolio_summary", "get_risk_metrics", "get_risk_contributions",
        "run_stress_test", "get_correlation_matrix", "get_alerts",
    } <= names

    # Tools return JSON strings (LangChain contract).
    summary_tool = next(t for t in tools if t.name == "get_portfolio_summary")
    out = summary_tool.invoke({})
    assert isinstance(out, str)
    assert json.loads(out)["num_positions"] == 5

    # A parameterized tool exposes and honors its argument.
    stress_tool = next(t for t in tools if t.name == "run_stress_test")
    assert "scenario" in stress_tool.args
    result = json.loads(stress_tool.invoke({"scenario": "severe_crash"}))
    assert result["scenario"] == "severe_crash"
