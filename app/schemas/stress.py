"""Pydantic schemas for stress testing."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field


class ScenarioInfo(BaseModel):
    """Metadata about an available stress scenario."""

    name: str
    description: str


class CustomScenarioRequest(BaseModel):
    """A user-defined stress scenario (all shocks are signed fractions, e.g. -0.2)."""

    name: str = Field(default="custom", max_length=64)
    ticker_shocks: dict[str, float] = Field(default_factory=dict)
    class_shocks: dict[str, float] = Field(default_factory=dict)
    default_shock: float | None = None


class StressLegSchema(BaseModel):
    ticker: str
    shock: float
    price_before: Decimal
    price_after: Decimal
    value_before: Decimal
    value_after: Decimal
    pnl: Decimal
    pct_of_portfolio: float


class StressResult(BaseModel):
    """Outcome of running a stress scenario against a portfolio."""

    portfolio_id: int
    scenario_name: str
    description: str
    portfolio_value_before: Decimal
    portfolio_value_after: Decimal
    total_pnl: Decimal
    total_loss: Decimal = Field(..., description="Positive loss magnitude (0 if a gain).")
    pct_loss: float = Field(..., description="Signed return under the scenario.")
    legs: list[StressLegSchema] = Field(default_factory=list)
    worst_assets: list[str] = Field(default_factory=list)
    disclaimer: str = Field(
        default="Hypothetical scenario on synthetic/demo data. Not financial advice."
    )
