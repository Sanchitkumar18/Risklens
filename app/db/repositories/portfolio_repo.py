"""Repository for portfolios."""

from __future__ import annotations

from sqlalchemy import select

from app.db.models import Portfolio
from app.db.repositories.base import BaseRepository


class PortfolioRepository(BaseRepository[Portfolio]):
    model = Portfolio

    def create(self, name: str, description: str | None = None) -> Portfolio:
        """Create and persist a new portfolio."""
        return self.add(Portfolio(name=name, description=description))

    def get_by_name(self, name: str) -> Portfolio | None:
        """Return a portfolio by its unique name, or ``None``."""
        stmt = select(Portfolio).where(Portfolio.name == name)
        return self.session.execute(stmt).scalar_one_or_none()
