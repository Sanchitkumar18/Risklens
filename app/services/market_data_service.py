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
from app.pipelines.cleaning import clean_market_data
from app.pipelines.ingestion import dataframe_to_rows
from app.pipelines.validation import validate_market_data
from app.schemas.market_data import IngestionSummary
from app.schemas.validation import ValidationReport

logger = get_logger("risklens.market_data")


class MarketDataService:
    """Application service for importing and querying market data."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.repo = MarketDataRepository(session)

    def ingest_dataframe(
        self, df: pd.DataFrame, *, validate: bool = True, commit: bool = True
    ) -> IngestionSummary:
        """Run the pipeline (parse → validate → clean → upsert) for a frame.

        With ``validate=True`` (default), rows failing hard checks are quarantined and
        reported rather than written; the remaining rows are deduplicated/sorted before
        upsert. Ingestion is idempotent — re-importing updates rows in place.
        """
        rows_read = len(df)
        report: ValidationReport | None = None

        if validate:
            result = validate_market_data(df)
            report = result.report
            frame = result.accepted
            if not frame.empty:
                frame, _ = clean_market_data(frame)
        else:
            frame = df

        rows = dataframe_to_rows(frame) if len(frame) else []
        written = self.repo.bulk_upsert(rows)
        if commit:
            self.session.commit()

        tickers = sorted({row["ticker"] for row in rows})
        rejected = report.rows_rejected if report else 0
        logger.info(
            "market data ingested",
            extra={
                "rows_read": rows_read,
                "rows_written": written,
                "rows_rejected": rejected,
                "tickers": tickers,
            },
        )
        return IngestionSummary(
            rows_read=rows_read,
            rows_written=written,
            rows_rejected=rejected,
            tickers=tickers,
            validation=report,
            message=(
                f"Ingested {written} row(s) across {len(tickers)} ticker(s); "
                f"{rejected} row(s) rejected."
            ),
        )

    def ingest_csv(
        self, path: str, *, validate: bool = True, commit: bool = True
    ) -> IngestionSummary:
        """Load a CSV file and ingest it through the pipeline."""
        from app.pipelines.ingestion import load_csv

        df = load_csv(path)
        return self.ingest_dataframe(df, validate=validate, commit=commit)
