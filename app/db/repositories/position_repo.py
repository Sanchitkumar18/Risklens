"""Repository for portfolio positions."""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy import select

from app.db.models import Position
from app.db.repositories.base import BaseRepository


class PositionRepository(BaseRepository[Position]):
    model = Position

    def list_by_portfolio(self, portfolio_id: int) -> list[Position]:
        """Return all positions for a portfolio, ordered by ticker."""
        stmt = (
            select(Position)
            .where(Position.portfolio_id == portfolio_id)
            .order_by(Position.ticker)
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_by_ticker(self, portfolio_id: int, ticker: str) -> Position | None:
        """Return the position for a given portfolio+ticker, or ``None``."""
        stmt = select(Position).where(
            Position.portfolio_id == portfolio_id, Position.ticker == ticker
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def upsert(
        self,
        portfolio_id: int,
        ticker: str,
        quantity: Decimal,
        average_price: Decimal,
    ) -> Position:
        """Create the position, or update quantity/price if it already exists.

        Enforces the one-row-per-(portfolio, ticker) invariant at the application
        level (the DB unique constraint is the backstop).
        """
        existing = self.get_by_ticker(portfolio_id, ticker)
        if existing is not None:
            existing.quantity = quantity
            existing.average_price = average_price
            self.session.flush()
            return existing
        return self.add(
            Position(
                portfolio_id=portfolio_id,
                ticker=ticker,
                quantity=quantity,
                average_price=average_price,
            )
        )
