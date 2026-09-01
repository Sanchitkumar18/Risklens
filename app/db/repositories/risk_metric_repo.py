"""Repository for risk-metric snapshots (append-only history)."""

from __future__ import annotations

from sqlalchemy import select

from app.db.models import RiskMetric
from app.db.repositories.base import BaseRepository


class RiskMetricRepository(BaseRepository[RiskMetric]):
    model = RiskMetric

    def list_by_portfolio(self, portfolio_id: int, limit: int | None = None) -> list[RiskMetric]:
        """Return snapshots for a portfolio, newest first."""
        stmt = (
            select(RiskMetric)
            .where(RiskMetric.portfolio_id == portfolio_id)
            .order_by(RiskMetric.calculation_date.desc(), RiskMetric.id.desc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def latest_for_portfolio(self, portfolio_id: int) -> RiskMetric | None:
        """Return the most recent snapshot for a portfolio, or ``None``."""
        results = self.list_by_portfolio(portfolio_id, limit=1)
        return results[0] if results else None
