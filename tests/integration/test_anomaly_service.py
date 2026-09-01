"""Integration tests for AnomalyService."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.pipelines.synthetic_data import generate_market_data
from app.schemas.portfolio import PortfolioCreate, PositionCreate
from app.services.anomaly_service import AnomalyService
from app.services.market_data_service import MarketDataService
from app.services.portfolio_service import PortfolioService


@pytest.fixture()
def seeded(db_session):
    # Two years of data; the generator injects crashes + single-name jumps.
    MarketDataService(db_session).ingest_dataframe(
        generate_market_data(start="2021-01-04", end="2022-12-30", seed=7)
    )
    return db_session


@pytest.mark.integration
def test_scan_tickers_finds_anomalies(seeded):
    svc = AnomalyService(seeded)
    result = svc.scan_tickers(["NVDA", "AAPL", "SPY"], contamination=0.02)
    assert result.rows_analyzed > 0
    assert result.anomalies_found > 0
    # Records are well-formed and sorted by score (desc).
    top = result.anomalies[0]
    assert top.ticker in {"NVDA", "AAPL", "SPY"}
    assert top.anomaly_type
    scores = [a.anomaly_score for a in result.anomalies]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.integration
def test_scan_portfolio_persists(seeded):
    ps = PortfolioService(seeded)
    pf = ps.create_portfolio(PortfolioCreate(name="P"))
    for t in ["NVDA", "AAPL"]:
        ps.add_position(pf.id, PositionCreate(ticker=t, quantity=Decimal(10), average_price=Decimal(100)))

    svc = AnomalyService(seeded)
    result = svc.scan_portfolio(pf.id, contamination=0.03)
    stored = svc.anomalies.list_by_portfolio(pf.id)
    assert len(stored) == result.anomalies_found > 0
    assert all(a.portfolio_id == pf.id for a in stored)


@pytest.mark.integration
def test_rescan_is_idempotent(seeded):
    ps = PortfolioService(seeded)
    pf = ps.create_portfolio(PortfolioCreate(name="P"))
    ps.add_position(pf.id, PositionCreate(ticker="NVDA", quantity=Decimal(10), average_price=Decimal(100)))

    svc = AnomalyService(seeded)
    first = svc.scan_portfolio(pf.id, contamination=0.03)
    second = svc.scan_portfolio(pf.id, contamination=0.03)
    assert first.anomalies_found == second.anomalies_found
    assert len(svc.anomalies.list_by_portfolio(pf.id)) == second.anomalies_found


@pytest.mark.integration
def test_higher_contamination_finds_more(seeded):
    svc = AnomalyService(seeded)
    low = svc.scan_tickers(["NVDA"], contamination=0.02, persist=False)
    high = svc.scan_tickers(["NVDA"], contamination=0.08, persist=False)
    assert high.anomalies_found > low.anomalies_found
