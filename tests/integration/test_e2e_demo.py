"""Consolidated end-to-end demo test — the full RiskLens workflow in one flow.

Mirrors the documented demo: load data → build portfolio → compute risk → detect
anomalies → raise alerts → run a stress scenario → ask the grounded assistant, and
asserts the numbers are consistent across layers (the assistant's VaR equals the risk
engine's VaR).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.pipelines.synthetic_data import generate_market_data
from app.risk.stress_testing import MARKET_CRASH, TECH_SELLOFF
from app.schemas.assistant import AssistantQuery
from app.schemas.portfolio import PortfolioCreate, PositionCreate
from app.services.alert_service import AlertService
from app.services.anomaly_service import AnomalyService
from app.services.assistant_service import AssistantService
from app.services.market_data_service import MarketDataService
from app.services.portfolio_service import PortfolioService
from app.services.risk_service import RiskService
from app.services.stress_service import StressService

DEMO_POSITIONS = [("AAPL", 100, 150), ("MSFT", 80, 250), ("NVDA", 50, 200), ("AMZN", 60, 130), ("GOOGL", 40, 120)]


@pytest.mark.integration
def test_full_demo_workflow(db_session):
    # 1. Load sample market data through the validation pipeline.
    ingest = MarketDataService(db_session).ingest_dataframe(generate_market_data(seed=42))
    assert ingest.rows_written > 10_000
    assert ingest.rows_rejected == 0

    # 2–3. Create the Tech Growth portfolio with positions.
    ps = PortfolioService(db_session)
    pf = ps.create_portfolio(PortfolioCreate(name="Tech Growth"))
    for t, q, ap in DEMO_POSITIONS:
        ps.add_position(pf.id, PositionCreate(ticker=t, quantity=Decimal(q), average_price=Decimal(ap)))

    val = ps.value_portfolio(pf.id)
    assert val.total_value > 0
    assert pytest.approx(sum(val.weights.values()), abs=1e-6) == 1.0

    # 4. Compute risk metrics (and persist a snapshot).
    risk = RiskService(db_session)
    report = risk.compute_metrics(pf.id, confidence_level=0.99)
    assert report.observations > 100
    assert report.risk_contributions[0].ticker == "NVDA"  # top risk driver
    assert risk.risk_metrics.latest_for_portfolio(pf.id) is not None

    # 5. Detect anomalies.
    anomalies = AnomalyService(db_session).scan_portfolio(pf.id, contamination=0.02)
    assert anomalies.anomalies_found >= 0

    # 6. Generate alerts.
    alerts = AlertService(db_session).evaluate(pf.id)
    assert alerts.breaches > 0
    assert alerts.created == alerts.breaches

    # 7. Run stress scenarios.
    crash = StressService(db_session).run_builtin(pf.id, MARKET_CRASH)
    assert crash.pct_loss == pytest.approx(-0.20, abs=1e-3)
    tech = StressService(db_session).run_builtin(pf.id, TECH_SELLOFF)
    assert tech.total_loss > 0

    # 8. Ask the grounded assistant — its numbers must match the engine.
    asst = AssistantService(db_session)
    summary = asst.query(AssistantQuery(portfolio_id=pf.id, question="Why is my portfolio risky right now?"))
    assert summary.intent == "risk_summary"
    assert "NVDA" in summary.answer
    assert summary.warnings == []

    var_answer = asst.query(
        AssistantQuery(portfolio_id=pf.id, question="Explain my 99% VaR.", confidence=0.99)
    )
    v99 = next(v for v in var_answer.grounded_data["get_risk_metrics"]["var_historical"] if v["confidence_level"] == 0.99)
    # Cross-layer consistency: the assistant states exactly the engine's VaR.
    engine_v99 = next(v for v in report.var_historical if v.confidence_level == 0.99)
    assert v99["var_value"] == pytest.approx(engine_v99.var_value, rel=1e-6)
    assert f"${v99['var_value']:,.0f}" in var_answer.answer
