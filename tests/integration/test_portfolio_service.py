"""Integration tests for PortfolioService (CRUD + valuation)."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from app.core.exceptions import (
    DuplicateResourceError,
    InvalidPosition,
    MarketDataNotFound,
    PortfolioNotFound,
    PositionNotFound,
)
from app.schemas.portfolio import PortfolioCreate, PositionCreate, PositionUpdate
from app.services.market_data_service import MarketDataService
from app.services.portfolio_service import PortfolioService


def _bars(prices: dict[str, list[float]], start: str = "2024-01-02") -> pd.DataFrame:
    """Build a small OHLCV frame; the LAST date carries the given prices."""
    rows = []
    for ticker, series in prices.items():
        dates = pd.bdate_range(start, periods=len(series))
        for d, p in zip(dates, series, strict=True):
            rows.append(
                {
                    "date": d.date().isoformat(), "ticker": ticker,
                    "open": p, "high": p * 1.01, "low": p * 0.99,
                    "close": p, "adjusted_close": p, "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def seeded(db_session):
    """DB with AAPL @100 and MSFT @200 as their latest prices."""
    MarketDataService(db_session).ingest_dataframe(
        _bars({"AAPL": [98.0, 99.0, 100.0], "MSFT": [190.0, 195.0, 200.0]})
    )
    return PortfolioService(db_session)


# ── CRUD ────────────────────────────────────────────────────
@pytest.mark.integration
def test_create_and_get_portfolio(seeded):
    pf = seeded.create_portfolio(PortfolioCreate(name="Tech Growth", description="demo"))
    assert seeded.get_portfolio(pf.id).name == "Tech Growth"


@pytest.mark.integration
def test_duplicate_name_rejected(seeded):
    seeded.create_portfolio(PortfolioCreate(name="Dup"))
    with pytest.raises(DuplicateResourceError):
        seeded.create_portfolio(PortfolioCreate(name="Dup"))


@pytest.mark.integration
def test_get_missing_portfolio_raises(seeded):
    with pytest.raises(PortfolioNotFound):
        seeded.get_portfolio(9999)


@pytest.mark.integration
def test_add_position_invalid_ticker(seeded):
    pf = seeded.create_portfolio(PortfolioCreate(name="P"))
    with pytest.raises(InvalidPosition):
        seeded.add_position(pf.id, PositionCreate(ticker="1BAD$", quantity=Decimal("1"), average_price=Decimal("10")))


@pytest.mark.integration
def test_add_position_zero_quantity(seeded):
    pf = seeded.create_portfolio(PortfolioCreate(name="P"))
    with pytest.raises(InvalidPosition):
        seeded.add_position(pf.id, PositionCreate(ticker="AAPL", quantity=Decimal("0"), average_price=Decimal("10")))


@pytest.mark.integration
def test_add_position_upserts(seeded):
    pf = seeded.create_portfolio(PortfolioCreate(name="P"))
    seeded.add_position(pf.id, PositionCreate(ticker="AAPL", quantity=Decimal("10"), average_price=Decimal("90")))
    seeded.add_position(pf.id, PositionCreate(ticker="AAPL", quantity=Decimal("15"), average_price=Decimal("95")))
    holdings = seeded.get_holdings(pf.id)
    assert len(holdings) == 1
    assert holdings[0].quantity == Decimal("15")


@pytest.mark.integration
def test_update_and_remove_position(seeded):
    pf = seeded.create_portfolio(PortfolioCreate(name="P"))
    pos = seeded.add_position(pf.id, PositionCreate(ticker="AAPL", quantity=Decimal("10"), average_price=Decimal("90")))

    seeded.update_position(pf.id, pos.id, PositionUpdate(quantity=Decimal("20")))
    assert seeded.get_holdings(pf.id)[0].quantity == Decimal("20")

    seeded.remove_position(pf.id, pos.id)
    assert seeded.get_holdings(pf.id) == []


@pytest.mark.integration
def test_remove_position_wrong_portfolio(seeded):
    pf1 = seeded.create_portfolio(PortfolioCreate(name="P1"))
    pf2 = seeded.create_portfolio(PortfolioCreate(name="P2"))
    pos = seeded.add_position(pf1.id, PositionCreate(ticker="AAPL", quantity=Decimal("1"), average_price=Decimal("90")))
    with pytest.raises(PositionNotFound):
        seeded.remove_position(pf2.id, pos.id)


# ── Valuation ───────────────────────────────────────────────
@pytest.mark.integration
def test_valuation_math(seeded):
    pf = seeded.create_portfolio(PortfolioCreate(name="Tech Growth"))
    seeded.add_position(pf.id, PositionCreate(ticker="AAPL", quantity=Decimal("10"), average_price=Decimal("90")))
    seeded.add_position(pf.id, PositionCreate(ticker="MSFT", quantity=Decimal("5"), average_price=Decimal("210")))

    val = seeded.value_portfolio(pf.id)
    # AAPL: 10*100=1000, MSFT: 5*200=1000
    assert val.total_value == Decimal("2000.00")
    assert val.gross_exposure == Decimal("2000.00")
    assert val.net_exposure == Decimal("2000.00")
    assert val.weights["AAPL"] == pytest.approx(0.5)
    assert val.weights["MSFT"] == pytest.approx(0.5)
    # cost: 900 + 1050 = 1950 → unrealized 2000-1950 = 50
    assert val.total_cost == Decimal("1950.00")
    assert val.unrealized_pnl == Decimal("50.00")


@pytest.mark.integration
def test_empty_portfolio_values_to_zero(seeded):
    pf = seeded.create_portfolio(PortfolioCreate(name="Empty"))
    val = seeded.value_portfolio(pf.id)
    assert val.total_value == Decimal("0")
    assert val.holdings == []
    assert val.weights == {}


@pytest.mark.integration
def test_single_asset_weight_is_one(seeded):
    pf = seeded.create_portfolio(PortfolioCreate(name="Solo"))
    seeded.add_position(pf.id, PositionCreate(ticker="AAPL", quantity=Decimal("3"), average_price=Decimal("90")))
    val = seeded.value_portfolio(pf.id)
    assert val.weights["AAPL"] == pytest.approx(1.0)


@pytest.mark.integration
def test_missing_market_data_raises(seeded):
    pf = seeded.create_portfolio(PortfolioCreate(name="P"))
    seeded.add_position(pf.id, PositionCreate(ticker="TSLA", quantity=Decimal("1"), average_price=Decimal("100")))
    with pytest.raises(MarketDataNotFound):
        seeded.value_portfolio(pf.id)


@pytest.mark.integration
def test_long_short_gross_vs_net(seeded):
    pf = seeded.create_portfolio(PortfolioCreate(name="LS"))
    seeded.add_position(pf.id, PositionCreate(ticker="AAPL", quantity=Decimal("10"), average_price=Decimal("90")))   # +1000
    seeded.add_position(pf.id, PositionCreate(ticker="MSFT", quantity=Decimal("-5"), average_price=Decimal("210")))  # -1000
    val = seeded.value_portfolio(pf.id)
    assert val.gross_exposure == Decimal("2000.00")
    assert val.net_exposure == Decimal("0.00")
    assert val.weights["AAPL"] == pytest.approx(0.5)
    assert val.weights["MSFT"] == pytest.approx(-0.5)
