"""Pydantic schemas for portfolios, positions, holdings, and valuation."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


# ── Requests ────────────────────────────────────────────────
class PortfolioCreate(BaseModel):
    """Payload to create a portfolio."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2000)


class PositionCreate(BaseModel):
    """Payload to add/set a position."""

    ticker: str = Field(..., min_length=1, max_length=16)
    quantity: Decimal = Field(..., description="Signed share count (negative = short).")
    average_price: Decimal = Field(..., gt=0, description="Average cost per share.")


class PositionUpdate(BaseModel):
    """Partial update for an existing position."""

    quantity: Decimal | None = Field(default=None)
    average_price: Decimal | None = Field(default=None, gt=0)


# ── Responses ───────────────────────────────────────────────
class PositionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ticker: str
    quantity: Decimal
    average_price: Decimal
    created_at: datetime | None = None


class PortfolioRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None = None
    created_at: datetime | None = None
    positions: list[PositionRead] = Field(default_factory=list)


class Holding(BaseModel):
    """A position marked to market, with weight and unrealized P&L."""

    ticker: str
    quantity: Decimal
    average_price: Decimal
    last_price: Decimal
    price_date: date
    market_value: Decimal
    cost_basis: Decimal
    unrealized_pnl: Decimal
    weight: float = Field(..., description="Share of gross exposure (signed).")


class PortfolioValuation(BaseModel):
    """A full mark-to-market snapshot of a portfolio."""

    portfolio_id: int
    name: str
    as_of_date: date | None
    total_value: Decimal = Field(..., description="Net market value (Σ qty·price).")
    total_cost: Decimal
    unrealized_pnl: Decimal
    gross_exposure: Decimal = Field(..., description="Σ |market value|.")
    net_exposure: Decimal = Field(..., description="Σ market value (signed).")
    holdings: list[Holding] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
