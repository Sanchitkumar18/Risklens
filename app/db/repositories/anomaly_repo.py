"""Repository for detected anomalies."""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select

from app.db.models import Anomaly
from app.db.repositories.base import BaseRepository


class AnomalyRepository(BaseRepository[Anomaly]):
    model = Anomaly

    def bulk_add(self, anomalies: list[dict[str, Any]]) -> int:
        """Insert many anomaly rows in one flush. Returns the count inserted."""
        if not anomalies:
            return 0
        self.session.add_all([Anomaly(**row) for row in anomalies])
        self.session.flush()
        return len(anomalies)

    def list_by_portfolio(self, portfolio_id: int, limit: int | None = None) -> list[Anomaly]:
        """Return anomalies linked to a portfolio, most recent first."""
        stmt = (
            select(Anomaly)
            .where(Anomaly.portfolio_id == portfolio_id)
            .order_by(Anomaly.date.desc(), Anomaly.id.desc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def delete_by_portfolio(self, portfolio_id: int) -> int:
        """Delete all anomalies linked to a portfolio (idempotent re-scan). Returns count."""
        result = self.session.execute(
            delete(Anomaly).where(Anomaly.portfolio_id == portfolio_id)
        )
        self.session.flush()
        return int(result.rowcount or 0)

    def list_by_ticker(self, ticker: str, limit: int | None = None) -> list[Anomaly]:
        """Return anomalies for a ticker, most recent first."""
        stmt = (
            select(Anomaly)
            .where(Anomaly.ticker == ticker)
            .order_by(Anomaly.date.desc(), Anomaly.id.desc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())
