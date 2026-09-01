"""Pydantic schemas for the risk report and its components."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class VarResult(BaseModel):
    """VaR at one confidence level (loss is positive)."""

    confidence_level: float
    method: str = Field(..., description="'historical' or 'parametric'.")
    var_fraction: float = Field(..., description="Loss as a fraction of portfolio value.")
    var_value: float = Field(..., description="Loss in currency units.")


class DrawdownResultSchema(BaseModel):
    max_drawdown: float = Field(..., description="Worst peak-to-trough loss (negative fraction).")
    peak_date: date | None = None
    trough_date: date | None = None


class CorrelationPair(BaseModel):
    ticker_a: str
    ticker_b: str
    correlation: float


class RiskContributionSchema(BaseModel):
    ticker: str
    weight: float
    marginal: float
    component: float
    percent: float = Field(..., description="Share of total portfolio volatility (0..1).")


class RiskReport(BaseModel):
    """Complete risk snapshot for a portfolio."""

    portfolio_id: int
    name: str
    as_of_date: date | None
    observations: int = Field(..., description="Number of return observations used.")
    confidence_level: float

    portfolio_value: float
    gross_exposure: float
    net_exposure: float

    volatility_daily: float
    volatility_annualized: float

    var_historical: list[VarResult] = Field(default_factory=list)
    var_parametric: VarResult | None = None

    drawdown: DrawdownResultSchema
    weights: dict[str, float] = Field(default_factory=dict)
    correlation_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    high_correlation_pairs: list[CorrelationPair] = Field(default_factory=list)
    risk_contributions: list[RiskContributionSchema] = Field(default_factory=list)

    disclaimer: str = Field(
        default=(
            "Analytics on synthetic/demo data for research purposes only. "
            "VaR is not a maximum-loss guarantee. Not financial advice."
        )
    )
