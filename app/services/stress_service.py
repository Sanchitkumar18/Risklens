"""Stress-testing service.

Composes portfolio valuation with the pure stress engine: it marks the book to
market, computes per-asset volatilities (for volatility-shock scenarios), resolves the
scenario's shocks, and returns the P&L outcome. All figures derive from stored
positions and prices.
"""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy.orm import Session

from app.core.exceptions import InvalidPosition
from app.core.logging import get_logger
from app.db.repositories.market_data_repo import MarketDataRepository
from app.pipelines.transformation import to_price_matrix
from app.risk import returns as ret_mod
from app.risk import stress_testing as st
from app.schemas.stress import (
    CustomScenarioRequest,
    ScenarioInfo,
    StressLegSchema,
    StressResult,
)
from app.services.portfolio_service import PortfolioService

logger = get_logger("risklens.stress")


class StressService:
    """Run scenario stress tests against a portfolio."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.portfolios = PortfolioService(session)
        self.market_data = MarketDataRepository(session)

    def list_scenarios(self) -> list[ScenarioInfo]:
        """Return metadata for the built-in scenarios."""
        return [
            ScenarioInfo(name=s.name, description=s.description)
            for s in st.builtin_scenarios().values()
        ]

    def run_builtin(self, portfolio_id: int, scenario_name: str) -> StressResult:
        """Run a named built-in scenario."""
        scenarios = st.builtin_scenarios()
        spec = scenarios.get(scenario_name)
        if spec is None:
            raise InvalidPosition(
                f"Unknown scenario {scenario_name!r}.",
                details={"available": sorted(scenarios)},
            )
        return self._run(portfolio_id, spec)

    def run_custom(self, portfolio_id: int, request: CustomScenarioRequest) -> StressResult:
        """Run a user-defined scenario."""
        spec = st.ScenarioSpec(
            name=request.name,
            description="Custom user-defined scenario.",
            ticker_shocks=request.ticker_shocks,
            class_shocks=request.class_shocks,
            default_shock=request.default_shock,
        )
        return self._run(portfolio_id, spec)

    # ── internals ───────────────────────────────────────────
    def _run(self, portfolio_id: int, spec: st.ScenarioSpec) -> StressResult:
        valuation = self.portfolios.value_portfolio(portfolio_id)
        holdings = [
            st.StressHolding(ticker=h.ticker, quantity=Decimal(h.quantity), price=Decimal(h.last_price))
            for h in valuation.holdings
        ]

        sigma_map = self._sigma_map([h.ticker for h in holdings]) if spec.vol_multiple else {}
        shocks = st.resolve_shocks(spec, [h.ticker for h in holdings], sigma_map=sigma_map)
        outcome = st.apply_stress(holdings, shocks, scenario_name=spec.name, description=spec.description)

        logger.info(
            "stress test run",
            extra={
                "portfolio_id": portfolio_id,
                "scenario": spec.name,
                "pct_loss": round(outcome.pct_loss, 4),
                "total_pnl": str(outcome.total_pnl),
            },
        )
        return self._to_schema(portfolio_id, outcome)

    def _sigma_map(self, tickers: list[str]) -> dict[str, float]:
        """Per-asset daily volatility from stored history (for vol-shock scenarios)."""
        if not tickers:
            return {}
        bars = self.market_data.get_for_tickers(tickers)
        import pandas as pd

        frame = pd.DataFrame(
            {
                "date": [b.date for b in bars],
                "ticker": [b.ticker for b in bars],
                "adjusted_close": [float(b.adjusted_close) for b in bars],
            }
        )
        prices = to_price_matrix(frame)
        rets = ret_mod.asset_returns(prices[tickers])
        return {t: float(rets[t].std(ddof=1)) for t in tickers if t in rets.columns}

    @staticmethod
    def _to_schema(portfolio_id: int, outcome: st.StressOutcome) -> StressResult:
        return StressResult(
            portfolio_id=portfolio_id,
            scenario_name=outcome.scenario_name,
            description=outcome.description,
            portfolio_value_before=outcome.portfolio_value_before,
            portfolio_value_after=outcome.portfolio_value_after,
            total_pnl=outcome.total_pnl,
            total_loss=outcome.total_loss,
            pct_loss=outcome.pct_loss,
            legs=[
                StressLegSchema(
                    ticker=leg.ticker, shock=leg.shock,
                    price_before=leg.price_before, price_after=leg.price_after,
                    value_before=leg.value_before, value_after=leg.value_after,
                    pnl=leg.pnl, pct_of_portfolio=leg.pct_of_portfolio,
                )
                for leg in outcome.legs
            ],
            worst_assets=outcome.worst_assets,
        )
