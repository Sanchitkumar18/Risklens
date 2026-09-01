"""Portfolio management service.

Owns portfolio/position CRUD and mark-to-market valuation. All portfolio-level
figures (market value, weights, exposure, P&L) are computed from **stored positions
× stored market prices** — never hardcoded. This is the service the risk engine
(Phase 6) builds on.

Conventions
-----------
* Prices are marked with ``adjusted_close`` (consistent with the risk engine's
  return series). In the synthetic dataset ``adjusted_close == close``.
* Money quantities stay ``Decimal`` (exact); ratios/weights are ``float``.
* ``weight_i = market_value_i / gross_exposure`` (signed), so long/short books
  weight sensibly and weights of a long-only book sum to 1.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from app.core.exceptions import (
    DuplicateResourceError,
    InvalidPosition,
    MarketDataNotFound,
    PortfolioNotFound,
    PositionNotFound,
)
from app.core.logging import get_logger
from app.db.models import Portfolio, Position
from app.db.repositories.market_data_repo import MarketDataRepository
from app.db.repositories.portfolio_repo import PortfolioRepository
from app.db.repositories.position_repo import PositionRepository
from app.schemas.portfolio import (
    Holding,
    PortfolioCreate,
    PortfolioValuation,
    PositionCreate,
    PositionUpdate,
)

logger = get_logger("risklens.portfolio")

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,15}$")
_ZERO = Decimal("0")


class PortfolioService:
    """Application service for portfolio and position management + valuation."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.portfolios = PortfolioRepository(session)
        self.positions = PositionRepository(session)
        self.market_data = MarketDataRepository(session)

    # ── Portfolio CRUD ──────────────────────────────────────
    def create_portfolio(self, data: PortfolioCreate) -> Portfolio:
        """Create a portfolio; name must be unique."""
        if self.portfolios.get_by_name(data.name) is not None:
            raise DuplicateResourceError(
                f"A portfolio named {data.name!r} already exists.",
                details={"name": data.name},
            )
        pf = self.portfolios.create(name=data.name, description=data.description)
        self.session.commit()
        logger.info(
            "portfolio created", extra={"portfolio_id": pf.id, "portfolio_name": pf.name}
        )
        return pf

    def get_portfolio(self, portfolio_id: int) -> Portfolio:
        """Return a portfolio or raise :class:`PortfolioNotFound`."""
        pf = self.portfolios.get(portfolio_id)
        if pf is None:
            raise PortfolioNotFound(
                f"Portfolio {portfolio_id} not found.", details={"portfolio_id": portfolio_id}
            )
        return pf

    def list_portfolios(self) -> list[Portfolio]:
        """Return all portfolios."""
        return self.portfolios.list()

    def delete_portfolio(self, portfolio_id: int) -> None:
        """Delete a portfolio and its positions (cascade)."""
        pf = self.get_portfolio(portfolio_id)
        self.portfolios.delete(pf)
        self.session.commit()
        logger.info("portfolio deleted", extra={"portfolio_id": portfolio_id})

    # ── Position CRUD ───────────────────────────────────────
    def add_position(self, portfolio_id: int, data: PositionCreate) -> Position:
        """Add or replace a position (unique per portfolio+ticker)."""
        self.get_portfolio(portfolio_id)
        ticker = self._normalize_ticker(data.ticker)
        self._validate_quantity(data.quantity)
        if data.average_price <= _ZERO:
            raise InvalidPosition(
                "average_price must be positive.", details={"average_price": str(data.average_price)}
            )

        pos = self.positions.upsert(
            portfolio_id, ticker, Decimal(data.quantity), Decimal(data.average_price)
        )
        self.session.commit()
        logger.info(
            "position set",
            extra={"portfolio_id": portfolio_id, "ticker": ticker, "quantity": str(pos.quantity)},
        )
        return pos

    def update_position(
        self, portfolio_id: int, position_id: int, data: PositionUpdate
    ) -> Position:
        """Update quantity and/or average price of an existing position."""
        pos = self._get_owned_position(portfolio_id, position_id)
        if data.quantity is not None:
            self._validate_quantity(data.quantity)
            pos.quantity = Decimal(data.quantity)
        if data.average_price is not None:
            if data.average_price <= _ZERO:
                raise InvalidPosition("average_price must be positive.")
            pos.average_price = Decimal(data.average_price)
        self.session.flush()
        self.session.commit()
        return pos

    def remove_position(self, portfolio_id: int, position_id: int) -> None:
        """Remove a position from a portfolio."""
        pos = self._get_owned_position(portfolio_id, position_id)
        self.positions.delete(pos)
        self.session.commit()
        logger.info("position removed", extra={"portfolio_id": portfolio_id, "position_id": position_id})

    def get_holdings(self, portfolio_id: int) -> list[Position]:
        """Return the raw positions of a portfolio."""
        self.get_portfolio(portfolio_id)
        return self.positions.list_by_portfolio(portfolio_id)

    # ── Valuation ───────────────────────────────────────────
    def value_portfolio(
        self, portfolio_id: int, as_of: date | None = None
    ) -> PortfolioValuation:
        """Mark the portfolio to market and compute weights, exposure, and P&L.

        Uses the latest available price per ticker (on/before ``as_of``). An empty
        portfolio values to zero (not an error). If any held ticker has no market
        data, raises :class:`MarketDataNotFound` naming the missing tickers.
        """
        pf = self.get_portfolio(portfolio_id)
        positions = self.positions.list_by_portfolio(portfolio_id)

        if not positions:
            return PortfolioValuation(
                portfolio_id=pf.id, name=pf.name, as_of_date=as_of,
                total_value=_ZERO, total_cost=_ZERO, unrealized_pnl=_ZERO,
                gross_exposure=_ZERO, net_exposure=_ZERO, holdings=[], weights={},
            )

        tickers = [p.ticker for p in positions]
        bars = self.market_data.latest_bars(tickers, as_of=as_of)
        missing = sorted(t for t in tickers if t not in bars)
        if missing:
            raise MarketDataNotFound(
                "No market data for one or more held tickers.",
                details={"tickers": missing},
            )

        # First pass: market value + cost per position.
        raw: list[dict] = []
        gross = _ZERO
        net = _ZERO
        total_cost = _ZERO
        as_of_used: date | None = None
        for pos in positions:
            bar = bars[pos.ticker]
            price = Decimal(bar.adjusted_close)
            mv = (pos.quantity * price).quantize(Decimal("0.01"))
            cost = (pos.quantity * pos.average_price).quantize(Decimal("0.01"))
            gross += abs(mv)
            net += mv
            total_cost += cost
            as_of_used = bar.date if as_of_used is None else max(as_of_used, bar.date)
            raw.append({"pos": pos, "price": price, "price_date": bar.date, "mv": mv, "cost": cost})

        denom = gross if gross != _ZERO else _ZERO

        holdings: list[Holding] = []
        weights: dict[str, float] = {}
        for item in raw:
            pos: Position = item["pos"]
            mv: Decimal = item["mv"]
            weight = float(mv / denom) if denom != _ZERO else 0.0
            weights[pos.ticker] = weight
            holdings.append(
                Holding(
                    ticker=pos.ticker,
                    quantity=pos.quantity,
                    average_price=pos.average_price,
                    last_price=item["price"],
                    price_date=item["price_date"],
                    market_value=mv,
                    cost_basis=item["cost"],
                    unrealized_pnl=(mv - item["cost"]).quantize(Decimal("0.01")),
                    weight=weight,
                )
            )

        holdings.sort(key=lambda h: abs(h.market_value), reverse=True)
        return PortfolioValuation(
            portfolio_id=pf.id,
            name=pf.name,
            as_of_date=as_of_used,
            total_value=net.quantize(Decimal("0.01")),
            total_cost=total_cost.quantize(Decimal("0.01")),
            unrealized_pnl=(net - total_cost).quantize(Decimal("0.01")),
            gross_exposure=gross.quantize(Decimal("0.01")),
            net_exposure=net.quantize(Decimal("0.01")),
            holdings=holdings,
            weights=weights,
        )

    # ── Helpers ─────────────────────────────────────────────
    def _get_owned_position(self, portfolio_id: int, position_id: int) -> Position:
        pos = self.positions.get(position_id)
        if pos is None or pos.portfolio_id != portfolio_id:
            raise PositionNotFound(
                f"Position {position_id} not found in portfolio {portfolio_id}.",
                details={"portfolio_id": portfolio_id, "position_id": position_id},
            )
        return pos

    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        norm = ticker.strip().upper()
        if not _TICKER_RE.match(norm):
            raise InvalidPosition(f"Invalid ticker symbol: {ticker!r}", details={"ticker": ticker})
        return norm

    @staticmethod
    def _validate_quantity(quantity: Decimal) -> None:
        if Decimal(quantity) == _ZERO:
            raise InvalidPosition("Position quantity must be non-zero.")
