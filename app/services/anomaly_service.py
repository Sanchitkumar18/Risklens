"""Anomaly-detection service.

Loads market data, engineers features, runs the Isolation Forest detector, persists
detected anomalies, and returns a scan result. A portfolio scan is idempotent: prior
anomalies for that portfolio are cleared before new ones are written.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd
from sqlalchemy.orm import Session

from app.analytics.anomaly_detection import DEFAULT_FEATURES, detect_anomalies
from app.core.logging import get_logger
from app.db.repositories.anomaly_repo import AnomalyRepository
from app.db.repositories.market_data_repo import MarketDataRepository
from app.pipelines.transformation import enrich
from app.schemas.anomaly import AnomalyRead, AnomalyRecord, AnomalyScanResult
from app.services.portfolio_service import PortfolioService

logger = get_logger("risklens.anomaly")


class AnomalyService:
    """Detect and persist market/portfolio anomalies."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.market_data = MarketDataRepository(session)
        self.anomalies = AnomalyRepository(session)
        self.portfolios = PortfolioService(session)

    def scan_portfolio(
        self, portfolio_id: int, contamination: float = 0.02, *, persist: bool = True
    ) -> AnomalyScanResult:
        """Scan a portfolio's held tickers for anomalies."""
        holdings = self.portfolios.get_holdings(portfolio_id)
        tickers = [h.ticker for h in holdings]
        return self.scan_tickers(
            tickers, contamination=contamination, portfolio_id=portfolio_id, persist=persist
        )

    def list_persisted(self, portfolio_id: int, limit: int | None = None) -> list[AnomalyRead]:
        """Return stored anomalies for a portfolio (most recent first)."""
        self.portfolios.get_portfolio(portfolio_id)  # 404 if missing
        rows = self.anomalies.list_by_portfolio(portfolio_id, limit=limit)
        return [AnomalyRead.model_validate(r) for r in rows]

    def scan_tickers(
        self,
        tickers: list[str],
        contamination: float = 0.02,
        portfolio_id: int | None = None,
        *,
        persist: bool = True,
    ) -> AnomalyScanResult:
        """Scan a set of tickers; persist any anomalies (optionally linked to a portfolio)."""
        if not tickers:
            return AnomalyScanResult(
                portfolio_id=portfolio_id, tickers=[], rows_analyzed=0,
                anomalies_found=0, contamination=contamination, anomalies=[],
            )

        bars = self.market_data.get_for_tickers(tickers)
        frame = _bars_to_frame(bars)
        enriched = enrich(frame)

        detected = detect_anomalies(enriched, contamination=contamination)
        rows_analyzed = int(len(detected))
        anomaly_rows = detected[detected["is_anomaly"]].copy()

        records = [
            AnomalyRecord(
                ticker=row["ticker"],
                date=_as_date(row["date"]),
                anomaly_score=float(row["anomaly_score"]),
                anomaly_type=str(row["anomaly_type"]),
                features={f: _safe_float(row.get(f)) for f in DEFAULT_FEATURES},
            )
            for _, row in anomaly_rows.iterrows()
        ]

        if persist and portfolio_id is not None:
            self.anomalies.delete_by_portfolio(portfolio_id)
            self.anomalies.bulk_add(
                [
                    {
                        "portfolio_id": portfolio_id,
                        "ticker": r.ticker,
                        "date": r.date,
                        "anomaly_score": Decimal(str(round(r.anomaly_score, 8))),
                        "anomaly_type": r.anomaly_type,
                        "features": r.features,
                    }
                    for r in records
                ]
            )
            self.session.commit()

        logger.info(
            "anomaly scan complete",
            extra={
                "portfolio_id": portfolio_id,
                "tickers": tickers,
                "rows_analyzed": rows_analyzed,
                "anomalies_found": len(records),
            },
        )
        return AnomalyScanResult(
            portfolio_id=portfolio_id,
            tickers=sorted(set(tickers)),
            rows_analyzed=rows_analyzed,
            anomalies_found=len(records),
            contamination=contamination,
            anomalies=sorted(records, key=lambda r: r.anomaly_score, reverse=True),
        )


def _bars_to_frame(bars: list) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [b.date for b in bars],
            "ticker": [b.ticker for b in bars],
            "adjusted_close": [float(b.adjusted_close) for b in bars],
            "volume": [int(b.volume) for b in bars],
        }
    )


def _as_date(value) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    return value


def _safe_float(value) -> float:
    try:
        f = float(value)
        return f if pd.notna(f) else 0.0
    except (TypeError, ValueError):
        return 0.0
