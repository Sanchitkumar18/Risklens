"""Repository for OHLCV market data, including idempotent bulk upsert."""

from __future__ import annotations

from datetime import date as date_type
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.db.models import MarketData
from app.db.repositories.base import BaseRepository

# Columns overwritten when a (ticker, date) row already exists.
_UPSERT_COLUMNS = ("open", "high", "low", "close", "adjusted_close", "volume")


class MarketDataRepository(BaseRepository[MarketData]):
    model = MarketData

    def bulk_upsert(self, rows: list[dict[str, Any]]) -> int:
        """Insert or update many bars in one statement (idempotent on ticker+date).

        Uses dialect-native ``ON CONFLICT ... DO UPDATE`` so re-ingesting the same
        file is safe and cheap (single round trip, no N+1). Returns the number of
        rows submitted.
        """
        if not rows:
            return 0

        dialect = self.session.bind.dialect.name  # type: ignore[union-attr]
        insert = pg_insert if dialect == "postgresql" else sqlite_insert
        stmt = insert(MarketData).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker", "date"],
            set_={col: getattr(stmt.excluded, col) for col in _UPSERT_COLUMNS},
        )
        self.session.execute(stmt)
        self.session.flush()
        return len(rows)

    def get_by_ticker(
        self,
        ticker: str,
        start: date_type | None = None,
        end: date_type | None = None,
    ) -> list[MarketData]:
        """Return bars for a ticker within an optional inclusive date range."""
        stmt = select(MarketData).where(MarketData.ticker == ticker)
        if start is not None:
            stmt = stmt.where(MarketData.date >= start)
        if end is not None:
            stmt = stmt.where(MarketData.date <= end)
        stmt = stmt.order_by(MarketData.date)
        return list(self.session.execute(stmt).scalars().all())

    def get_for_tickers(
        self,
        tickers: list[str],
        start: date_type | None = None,
        end: date_type | None = None,
    ) -> list[MarketData]:
        """Return bars for several tickers (one query — avoids per-ticker N+1)."""
        if not tickers:
            return []
        stmt = select(MarketData).where(MarketData.ticker.in_(tickers))
        if start is not None:
            stmt = stmt.where(MarketData.date >= start)
        if end is not None:
            stmt = stmt.where(MarketData.date <= end)
        stmt = stmt.order_by(MarketData.ticker, MarketData.date)
        return list(self.session.execute(stmt).scalars().all())

    def distinct_tickers(self) -> list[str]:
        """Return the sorted set of tickers present in the table."""
        stmt = select(MarketData.ticker).distinct().order_by(MarketData.ticker)
        return list(self.session.execute(stmt).scalars().all())

    def count_for_ticker(self, ticker: str) -> int:
        """Return the number of bars stored for a ticker."""
        stmt = select(MarketData).where(MarketData.ticker == ticker)
        return len(list(self.session.execute(stmt).scalars().all()))
