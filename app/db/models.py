"""SQLAlchemy ORM models — the seven RiskLens tables.

Design choices (see docs/database-design.md, added later):

* **Money/prices use ``Numeric(18, 6)``** — never float. Binary floats cannot
  represent decimal cents exactly; ``Numeric`` maps to Python ``Decimal`` and to
  Postgres ``NUMERIC``, preserving precision.
* **Primary keys are ``BigInteger`` on Postgres but ``Integer`` on SQLite**
  (via ``with_variant``). SQLite only auto-increments an ``INTEGER PRIMARY KEY``,
  so this keeps the identical model runnable under the in-memory test database.
* **Uniqueness/constraints are enforced in the database**, not just the app:
  idempotent ingestion (``uq_market_data_ticker_date``), one row per portfolio
  asset (``uq_position_portfolio_ticker``), and alert de-duplication (``uq_alert_dedup``).
"""

from __future__ import annotations

from datetime import date as date_type
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import BigInteger as _BigInteger
from sqlalchemy import JSON as _JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base

# ── Reusable column types ───────────────────────────────────
# Auto-incrementing key: BIGINT on Postgres, INTEGER (rowid alias) on SQLite.
BigIntId = _BigInteger().with_variant(Integer, "sqlite")
# 6-dp decimal for all monetary and price quantities.
Money = Numeric(18, 6)
# JSONB on Postgres (indexable), plain JSON elsewhere (SQLite tests).
JsonType = _JSON().with_variant(JSONB, "postgresql")


class MarketData(Base):
    """A single daily OHLCV bar for one ticker (adjusted close included)."""

    __tablename__ = "market_data"
    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_market_data_ticker_date"),
        CheckConstraint("high >= low", name="ck_market_data_high_ge_low"),
        CheckConstraint(
            "open > 0 AND high > 0 AND low > 0 AND close > 0 AND adjusted_close > 0",
            name="ck_market_data_positive_prices",
        ),
        CheckConstraint("volume >= 0", name="ck_market_data_volume_nonneg"),
        Index("ix_market_data_ticker_date", "ticker", "date"),
    )

    id: Mapped[int] = mapped_column(BigIntId, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False)
    date: Mapped[date_type] = mapped_column(Date, nullable=False)
    open: Mapped[Decimal] = mapped_column(Money, nullable=False)
    high: Mapped[Decimal] = mapped_column(Money, nullable=False)
    low: Mapped[Decimal] = mapped_column(Money, nullable=False)
    close: Mapped[Decimal] = mapped_column(Money, nullable=False)
    adjusted_close: Mapped[Decimal] = mapped_column(Money, nullable=False)
    volume: Mapped[int] = mapped_column(_BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<MarketData {self.ticker} {self.date} close={self.close}>"


class Portfolio(Base):
    """A named collection of positions."""

    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(BigIntId, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    positions: Mapped[list["Position"]] = relationship(
        back_populates="portfolio",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Portfolio id={self.id} name={self.name!r}>"


class Position(Base):
    """A holding of ``quantity`` shares of ``ticker`` inside a portfolio."""

    __tablename__ = "positions"
    __table_args__ = (
        UniqueConstraint("portfolio_id", "ticker", name="uq_position_portfolio_ticker"),
        CheckConstraint("quantity <> 0", name="ck_position_quantity_nonzero"),
        Index("ix_position_portfolio_id", "portfolio_id"),
    )

    id: Mapped[int] = mapped_column(BigIntId, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(
        BigIntId, ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False
    )
    ticker: Mapped[str] = mapped_column(String(16), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Money, nullable=False)
    average_price: Mapped[Decimal] = mapped_column(Money, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    portfolio: Mapped["Portfolio"] = relationship(back_populates="positions")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Position pf={self.portfolio_id} {self.ticker} qty={self.quantity}>"


class RiskMetric(Base):
    """An append-only snapshot of a risk calculation run for a portfolio.

    Kept as a history table (not upserted) so the assistant can answer questions
    like "why did risk *increase*?" by comparing successive snapshots.
    """

    __tablename__ = "risk_metrics"
    __table_args__ = (
        Index("ix_risk_metrics_portfolio_date", "portfolio_id", "calculation_date"),
    )

    id: Mapped[int] = mapped_column(BigIntId, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(
        BigIntId, ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False
    )
    calculation_date: Mapped[date_type] = mapped_column(Date, nullable=False)
    confidence_level: Mapped[Decimal] = mapped_column(Numeric(4, 3), nullable=False)
    var: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    parametric_var: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    volatility_annualized: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    max_drawdown: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    exposure_gross: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    exposure_net: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    stress_loss: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    method: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Anomaly(Base):
    """A market/portfolio observation flagged by the anomaly detector."""

    __tablename__ = "anomalies"
    __table_args__ = (
        Index("ix_anomalies_ticker_date", "ticker", "date"),
        Index("ix_anomalies_portfolio_id", "portfolio_id"),
    )

    id: Mapped[int] = mapped_column(BigIntId, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int | None] = mapped_column(
        BigIntId, ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=True
    )
    ticker: Mapped[str] = mapped_column(String(16), nullable=False)
    date: Mapped[date_type] = mapped_column(Date, nullable=False)
    anomaly_score: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    anomaly_type: Mapped[str] = mapped_column(String(64), nullable=False)
    features: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Alert(Base):
    """A persisted risk-limit breach or notable event.

    ``dedup_key`` is unique so the alert engine cannot insert the same logical
    event twice (enforced by the database, not fragile application logic).
    """

    __tablename__ = "alerts"
    __table_args__ = (
        UniqueConstraint("dedup_key", name="uq_alert_dedup"),
        Index("ix_alerts_portfolio_ack", "portfolio_id", "acknowledged"),
    )

    id: Mapped[int] = mapped_column(BigIntId, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(
        BigIntId, ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False
    )
    alert_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    metric_value: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    threshold: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    dedup_key: Mapped[str] = mapped_column(String(255), nullable=False)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
