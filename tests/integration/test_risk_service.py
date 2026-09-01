"""Integration tests for RiskService (end-to-end over stored data)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.core.exceptions import InsufficientHistoricalData
from app.pipelines.synthetic_data import generate_market_data
from app.schemas.portfolio import PortfolioCreate, PositionCreate
from app.services.market_data_service import MarketDataService
from app.services.portfolio_service import PortfolioService
from app.services.risk_service import RiskService


@pytest.fixture()
def portfolio(db_session):
    """A 3-asset portfolio backed by ~1 year of market data."""
    MarketDataService(db_session).ingest_dataframe(
        generate_market_data(start="2022-01-03", end="2022-12-31", seed=7)
    )
    ps = PortfolioService(db_session)
    pf = ps.create_portfolio(PortfolioCreate(name="Tech Growth"))
    for t, q, ap in [("AAPL", 100, 150), ("MSFT", 80, 250), ("NVDA", 50, 200)]:
        ps.add_position(pf.id, PositionCreate(ticker=t, quantity=Decimal(q), average_price=Decimal(ap)))
    return pf.id


@pytest.mark.integration
def test_full_risk_report(db_session, portfolio):
    report = RiskService(db_session).compute_metrics(portfolio, confidence_level=0.95)

    assert report.observations >= 20
    assert report.portfolio_value > 0

    # VaR present at all standard levels and monotonic in confidence.
    levels = {v.confidence_level: v.var_value for v in report.var_historical}
    assert set(levels) >= {0.90, 0.95, 0.99}
    assert levels[0.90] <= levels[0.95] <= levels[0.99]
    assert levels[0.95] > 0
    assert report.var_parametric is not None

    # Volatility plausible; drawdown non-positive.
    assert 0.05 < report.volatility_annualized < 1.5
    assert report.drawdown.max_drawdown <= 0

    # Weights and risk contributions each sum to ~1.
    assert sum(report.weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert sum(c.percent for c in report.risk_contributions) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.integration
def test_risk_metric_persisted(db_session, portfolio):
    service = RiskService(db_session)
    report = service.compute_metrics(portfolio, confidence_level=0.95)

    latest = service.risk_metrics.latest_for_portfolio(portfolio)
    assert latest is not None
    headline = next(v for v in report.var_historical if v.confidence_level == 0.95)
    assert float(latest.var) == pytest.approx(headline.var_value, rel=1e-4)
    assert float(latest.exposure_gross) == pytest.approx(report.gross_exposure, rel=1e-4)


@pytest.mark.integration
def test_empty_portfolio_raises(db_session):
    ps = PortfolioService(db_session)
    pf = ps.create_portfolio(PortfolioCreate(name="Empty"))
    with pytest.raises(InsufficientHistoricalData):
        RiskService(db_session).compute_metrics(pf.id)


@pytest.mark.integration
def test_insufficient_history_raises(db_session):
    # Only ~7 business days of data → below the tail-estimation minimum.
    MarketDataService(db_session).ingest_dataframe(
        generate_market_data(start="2022-01-03", end="2022-01-11", seed=1)
    )
    ps = PortfolioService(db_session)
    pf = ps.create_portfolio(PortfolioCreate(name="Short"))
    ps.add_position(pf.id, PositionCreate(ticker="AAPL", quantity=Decimal(10), average_price=Decimal(150)))
    with pytest.raises(InsufficientHistoricalData):
        RiskService(db_session).compute_metrics(pf.id)
