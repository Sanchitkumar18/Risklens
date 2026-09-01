"""Integration tests for the AlertService (end-to-end)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.core.exceptions import AlertNotFound
from app.pipelines.synthetic_data import generate_market_data
from app.schemas.alert import AlertThresholds
from app.schemas.portfolio import PortfolioCreate, PositionCreate
from app.services.alert_service import AlertService
from app.services.market_data_service import MarketDataService
from app.services.portfolio_service import PortfolioService


@pytest.fixture()
def portfolio(db_session):
    MarketDataService(db_session).ingest_dataframe(generate_market_data(seed=42))
    ps = PortfolioService(db_session)
    pf = ps.create_portfolio(PortfolioCreate(name="Tech Growth"))
    for t, q, ap in [("AAPL", 100, 150), ("MSFT", 80, 250), ("NVDA", 50, 200), ("AMZN", 60, 130), ("GOOGL", 40, 120)]:
        ps.add_position(pf.id, PositionCreate(ticker=t, quantity=Decimal(q), average_price=Decimal(ap)))
    return pf.id


@pytest.mark.integration
def test_evaluate_generates_alerts(db_session, portfolio):
    result = AlertService(db_session).evaluate(portfolio)
    assert result.breaches > 0
    assert result.created == result.breaches      # first run: all new
    assert sum(result.by_severity.values()) == result.breaches
    # Persisted and retrievable.
    stored = AlertService(db_session).list_alerts(portfolio)
    assert len(stored) == result.created


@pytest.mark.integration
def test_evaluate_is_deduplicated(db_session, portfolio):
    svc = AlertService(db_session)
    first = svc.evaluate(portfolio)
    second = svc.evaluate(portfolio)          # same day → no new alerts
    assert second.created == 0
    assert second.existing == second.breaches
    assert len(svc.list_alerts(portfolio)) == first.created


@pytest.mark.integration
def test_tight_limits_raise_critical(db_session, portfolio):
    tight = AlertThresholds(
        var_limit=0.005, volatility_limit=0.05, drawdown_limit=0.05,
        max_single_weight=0.10, stress_loss_limit=0.05,
    )
    result = AlertService(db_session).evaluate(portfolio, tight, include_anomalies=False)
    assert result.by_severity.get("CRITICAL", 0) > 0


@pytest.mark.integration
def test_loose_limits_no_alerts(db_session, portfolio):
    loose = AlertThresholds(
        var_limit=0.99, volatility_limit=5.0, drawdown_limit=0.99,
        max_single_weight=0.99, anomaly_score_limit=99.0, stress_loss_limit=0.99,
    )
    result = AlertService(db_session).evaluate(portfolio, loose)
    assert result.breaches == 0


@pytest.mark.integration
def test_acknowledge(db_session, portfolio):
    svc = AlertService(db_session)
    svc.evaluate(portfolio)
    alerts = svc.list_alerts(portfolio, acknowledged=False)
    assert alerts

    svc.acknowledge(portfolio, alerts[0].id)
    assert len(svc.list_alerts(portfolio, acknowledged=True)) == 1
    assert len(svc.list_alerts(portfolio, acknowledged=False)) == len(alerts) - 1

    with pytest.raises(AlertNotFound):
        svc.acknowledge(portfolio, 999999)
