"""Integration tests for StressService."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.pipelines.synthetic_data import generate_market_data
from app.risk.stress_testing import MARKET_CRASH, TECH_SELLOFF, VOLATILITY_SHOCK
from app.schemas.portfolio import PortfolioCreate, PositionCreate
from app.schemas.stress import CustomScenarioRequest
from app.services.market_data_service import MarketDataService
from app.services.portfolio_service import PortfolioService
from app.services.stress_service import StressService


@pytest.fixture()
def portfolio(db_session):
    MarketDataService(db_session).ingest_dataframe(
        generate_market_data(start="2022-01-03", end="2022-12-31", seed=7)
    )
    ps = PortfolioService(db_session)
    pf = ps.create_portfolio(PortfolioCreate(name="Tech Growth"))
    for t, q, ap in [("AAPL", 100, 150), ("MSFT", 80, 250), ("SPY", 20, 400)]:
        ps.add_position(pf.id, PositionCreate(ticker=t, quantity=Decimal(q), average_price=Decimal(ap)))
    return pf.id


@pytest.mark.integration
def test_list_scenarios(db_session):
    names = {s.name for s in StressService(db_session).list_scenarios()}
    assert {MARKET_CRASH, TECH_SELLOFF, VOLATILITY_SHOCK} <= names


@pytest.mark.integration
def test_market_crash_loses_20pct(db_session, portfolio):
    res = StressService(db_session).run_builtin(portfolio, MARKET_CRASH)
    # −20% on every leg; tiny deviation from cent-level rounding of each leg's value.
    assert res.pct_loss == pytest.approx(-0.20, abs=1e-4)
    assert res.total_loss > 0
    assert all(leg.pnl < 0 for leg in res.legs)


@pytest.mark.integration
def test_tech_selloff_hits_tech_harder(db_session, portfolio):
    res = StressService(db_session).run_builtin(portfolio, TECH_SELLOFF)
    shocks = {leg.ticker: leg.shock for leg in res.legs}
    assert shocks["AAPL"] == pytest.approx(-0.25)
    assert shocks["MSFT"] == pytest.approx(-0.25)
    assert shocks["SPY"] == pytest.approx(-0.10)
    # Worst asset should be a tech name (larger shock × larger position).
    assert res.worst_assets[0] in {"AAPL", "MSFT"}


@pytest.mark.integration
def test_volatility_shock_produces_losses(db_session, portfolio):
    res = StressService(db_session).run_builtin(portfolio, VOLATILITY_SHOCK)
    assert res.pct_loss < 0
    # Each shock is a 3-sigma daily move → small but non-zero.
    assert all(leg.shock < 0 for leg in res.legs)


@pytest.mark.integration
def test_custom_scenario(db_session, portfolio):
    req = CustomScenarioRequest(
        name="my_scenario", ticker_shocks={"AAPL": -0.5}, default_shock=0.0
    )
    res = StressService(db_session).run_custom(portfolio, req)
    aapl = next(leg for leg in res.legs if leg.ticker == "AAPL")
    assert aapl.shock == -0.5
    spy = next(leg for leg in res.legs if leg.ticker == "SPY")
    assert spy.shock == 0.0
    assert res.worst_assets[0] == "AAPL"
