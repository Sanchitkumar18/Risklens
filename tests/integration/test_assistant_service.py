"""Integration tests for the LangGraph assistant (deterministic mock mode)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.pipelines.synthetic_data import generate_market_data
from app.schemas.assistant import AssistantQuery
from app.schemas.portfolio import PortfolioCreate, PositionCreate
from app.services.assistant_service import AssistantService
from app.services.market_data_service import MarketDataService
from app.services.portfolio_service import PortfolioService


@pytest.fixture()
def portfolio_id(db_session) -> int:
    MarketDataService(db_session).ingest_dataframe(generate_market_data(seed=42))
    ps = PortfolioService(db_session)
    pf = ps.create_portfolio(PortfolioCreate(name="Tech Growth"))
    for t, q, ap in [("AAPL", 100, 150), ("MSFT", 80, 250), ("NVDA", 50, 200), ("AMZN", 60, 130), ("GOOGL", 40, 120)]:
        ps.add_position(pf.id, PositionCreate(ticker=t, quantity=Decimal(q), average_price=Decimal(ap)))
    return pf.id


def _ask(db, pid, question, confidence=None):
    return AssistantService(db).query(
        AssistantQuery(portfolio_id=pid, question=question, confidence=confidence)
    )


@pytest.mark.integration
def test_var_explanation_is_grounded(db_session, portfolio_id):
    resp = _ask(db_session, portfolio_id, "Explain my 99% VaR in simple terms.", confidence=0.99)
    assert resp.intent == "var_explain"
    assert "get_risk_metrics" in resp.tools_used
    assert "VaR" in resp.answer and "99%" in resp.answer
    assert "not financial advice" in resp.answer.lower()
    assert resp.warnings == []
    # The dollar figure in the answer must equal a tool-produced number.
    v99 = next(v for v in resp.grounded_data["get_risk_metrics"]["var_historical"] if v["confidence_level"] == 0.99)
    assert f"${v99['var_value']:,.0f}" in resp.answer


@pytest.mark.integration
def test_contribution_names_top_driver(db_session, portfolio_id):
    resp = _ask(db_session, portfolio_id, "Which asset contributes most to portfolio risk?")
    assert resp.intent == "contribution"
    assert "NVDA" in resp.answer  # highest-vol name dominates risk


@pytest.mark.integration
def test_stress_question(db_session, portfolio_id):
    resp = _ask(db_session, portfolio_id, "What happens if technology stocks fall 25%?")
    assert resp.intent == "stress"
    assert "run_stress_test" in resp.tools_used
    assert resp.grounded_data["run_stress_test"]["scenario"] == "tech_selloff"
    assert "lose" in resp.answer.lower()


@pytest.mark.integration
def test_risk_summary(db_session, portfolio_id):
    resp = _ask(db_session, portfolio_id, "Why is my portfolio risky right now?")
    assert resp.intent == "risk_summary"
    assert "get_risk_metrics" in resp.tools_used
    assert "volatility" in resp.answer.lower()


@pytest.mark.integration
def test_alerts_question(db_session, portfolio_id):
    resp = _ask(db_session, portfolio_id, "What are my current risk alerts?")
    assert resp.intent == "alerts"
    assert "get_alerts" in resp.tools_used


@pytest.mark.integration
@pytest.mark.parametrize(
    "question,intent,tool",
    [
        ("Show my asset exposure and weights.", "exposure", "get_asset_exposure"),
        ("How correlated are my holdings?", "correlation", "get_correlation_matrix"),
        ("What caused the largest drawdown?", "drawdown", "get_drawdown_analysis"),
        ("Which assets have unusual movements?", "anomalies", "get_anomalies"),
    ],
)
def test_remaining_intents(db_session, portfolio_id, question, intent, tool):
    resp = _ask(db_session, portfolio_id, question)
    assert resp.intent == intent
    assert tool in resp.tools_used
    assert resp.answer


@pytest.mark.integration
def test_empty_portfolio_degrades_gracefully(db_session):
    ps = PortfolioService(db_session)
    pf = ps.create_portfolio(PortfolioCreate(name="Empty"))
    resp = _ask(db_session, pf.id, "Why is my portfolio risky?")
    # No crash; the assistant says it lacks data and warns.
    assert resp.answer
    assert resp.warnings  # tool errors surfaced


@pytest.mark.integration
def test_assistant_api_endpoint(client_db):
    from app.pipelines.synthetic_data import generate_market_data as gen
    import io

    df = gen(start="2022-01-03", end="2022-12-31", seed=7)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    client_db.post("/api/v1/market-data/upload", files={"file": ("d.csv", buf, "text/csv")})
    pid = client_db.post("/api/v1/portfolios", json={"name": "P"}).json()["id"]
    for t in ["AAPL", "MSFT", "NVDA"]:
        client_db.post(f"/api/v1/portfolios/{pid}/positions", json={"ticker": t, "quantity": "10", "average_price": "100"})

    resp = client_db.post(
        "/api/v1/assistant/query",
        json={"portfolio_id": pid, "question": "Explain my 95% VaR."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "var_explain"
    assert "VaR" in body["answer"]
