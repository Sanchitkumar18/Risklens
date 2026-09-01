"""Market-data service — orchestrates ingestion into the database.

Sits between the pure parsing pipeline (``app.pipelines.ingestion``) and the
repository. This is the seam where Phase 4's validation/cleaning/transformation
steps will be inserted (parse → validate → clean → transform → upsert) without
changing callers.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.repositories.market_data_repo import MarketDataRepository
from app.pipelines.ingestion import dataframe_to_rows
from app.schemas.market_data import IngestionSummary

logger = get_logger("risklens.market_data")


class MarketDataService:
    """Application service for importing and querying market data."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.repo = MarketDataRepository(session)

    def ingest_dataframe(self, df: pd.DataFrame, *, commit: bool = True) -> IngestionSummary:
        """Parse, upsert, and (optionally) commit a market-data frame.

        Returns an :class:`IngestionSummary`. Ingestion is idempotent — re-importing
        the same rows updates in place rather than duplicating (see the repository's
        ``bulk_upsert``).
        """
        rows = dataframe_to_rows(df)
        rows_read = len(rows)

        written = self.repo.bulk_upsert(rows)
        if commit:
            self.session.commit()

        tickers = sorted({row["ticker"] for row in rows})
        logger.info(
            "market data ingested",
            extra={"rows_read": rows_read, "rows_written": written, "tickers": tickers},
        )
        return IngestionSummary(
            rows_read=rows_read,
            rows_written=written,
            tickers=tickers,
            message=f"Ingested {written} row(s) across {len(tickers)} ticker(s).",
        )

    def ingest_csv(self, path: str, *, commit: bool = True) -> IngestionSummary:
        """Load a CSV file and ingest it."""
        from app.pipelines.ingestion import load_csv

        df = load_csv(path)
        return self.ingest_dataframe(df, commit=commit)
