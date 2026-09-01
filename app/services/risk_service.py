"""Risk service — orchestrates the risk engine over stored data.

Loads a portfolio's positions and price history, builds the portfolio return series
(current holdings × historical prices), computes every risk metric from the pure
``app.risk`` / ``app.analytics`` functions, persists a ``RiskMetric`` snapshot, and
returns a :class:`RiskReport`. No metric is ever hardcoded — everything derives from
the database.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from app.analytics.risk_contribution import risk_contributions
from app.core.config import get_settings
from app.core.exceptions import InsufficientHistoricalData
from app.core.logging import get_logger
from app.db.models import RiskMetric
from app.db.repositories.market_data_repo import MarketDataRepository
from app.db.repositories.risk_metric_repo import RiskMetricRepository
from app.pipelines.transformation import to_price_matrix
from app.risk import correlation as corr_mod
from app.risk import drawdown as dd_mod
from app.risk import returns as ret_mod
from app.risk import var as var_mod
from app.risk import volatility as vol_mod
from app.schemas.risk import (
    CorrelationPair,
    DrawdownResultSchema,
    RiskContributionSchema,
    RiskReport,
    VarResult,
)
from app.services.portfolio_service import PortfolioService

logger = get_logger("risklens.risk")

_STANDARD_CONFIDENCE_LEVELS = (0.90, 0.95, 0.99)
_MIN_OBSERVATIONS = 20  # enough to estimate a 95–99% tail quantile with some stability


class RiskService:
    """Compute and persist portfolio risk metrics."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.settings = get_settings()
        self.market_data = MarketDataRepository(session)
        self.risk_metrics = RiskMetricRepository(session)
        self.portfolios = PortfolioService(session)

    def compute_metrics(
        self,
        portfolio_id: int,
        confidence_level: float | None = None,
        *,
        persist: bool = True,
    ) -> RiskReport:
        """Compute the full risk report for a portfolio and optionally persist it."""
        confidence_level = confidence_level or self.settings.default_confidence_level

        valuation = self.portfolios.value_portfolio(portfolio_id)
        if not valuation.holdings:
            raise InsufficientHistoricalData(
                "Portfolio has no positions to assess.", details={"portfolio_id": portfolio_id}
            )

        tickers = [h.ticker for h in valuation.holdings]
        quantities = {h.ticker: float(h.quantity) for h in valuation.holdings}
        market_values = {h.ticker: float(h.market_value) for h in valuation.holdings}

        # ── Load price history → wide adjusted-close matrix ──
        bars = self.market_data.get_for_tickers(tickers)
        frame = _bars_to_frame(bars)
        prices = to_price_matrix(frame)

        values = ret_mod.build_portfolio_values(prices, quantities)
        port_returns = ret_mod.portfolio_returns(values)
        asset_ret = ret_mod.asset_returns(prices[tickers])

        n_obs = len(port_returns)
        if n_obs < _MIN_OBSERVATIONS:
            raise InsufficientHistoricalData(
                "Not enough overlapping observations to compute risk.",
                details={"observations": n_obs, "required": _MIN_OBSERVATIONS},
            )

        portfolio_value = float(values.iloc[-1])
        as_of = _as_date(values.index[-1])

        # ── Metrics ─────────────────────────────────────────
        vol_daily = vol_mod.daily_volatility(port_returns)
        vol_annual = vol_mod.annualized_volatility(
            port_returns, periods_per_year=self.settings.trading_days_per_year
        )

        levels = sorted({*_STANDARD_CONFIDENCE_LEVELS, confidence_level})
        var_hist = [
            VarResult(
                confidence_level=cl,
                method="historical",
                var_fraction=(vf := var_mod.historical_var(port_returns, cl)),
                var_value=var_mod.var_to_dollars(vf, portfolio_value),
            )
            for cl in levels
        ]
        p_frac = var_mod.parametric_var(port_returns, confidence_level)
        var_param = VarResult(
            confidence_level=confidence_level,
            method="parametric",
            var_fraction=p_frac,
            var_value=var_mod.var_to_dollars(p_frac, portfolio_value),
        )

        dd = dd_mod.max_drawdown_detail(values)

        corr = corr_mod.correlation_matrix(asset_ret)
        pairs = corr_mod.high_correlation_pairs(corr, threshold=0.8)

        # Net weights for volatility decomposition (sum to 1 for long-only).
        from app.risk.exposure import net_weights

        contrib = risk_contributions(asset_ret, net_weights(market_values))

        # Headline (requested confidence) historical VaR for persistence.
        headline = next(v for v in var_hist if v.confidence_level == confidence_level)

        if persist:
            self._persist(
                portfolio_id=portfolio_id,
                as_of=as_of,
                confidence_level=confidence_level,
                var_value=headline.var_value,
                parametric_var_value=var_param.var_value,
                vol_annual=vol_annual,
                max_drawdown=dd.max_drawdown,
                gross=float(valuation.gross_exposure),
                net=float(valuation.net_exposure),
            )

        logger.info(
            "risk computed",
            extra={
                "portfolio_id": portfolio_id,
                "observations": n_obs,
                "var_value": round(headline.var_value, 2),
                "vol_annualized": round(vol_annual, 4),
            },
        )

        return RiskReport(
            portfolio_id=portfolio_id,
            name=valuation.name,
            as_of_date=as_of,
            observations=n_obs,
            confidence_level=confidence_level,
            portfolio_value=portfolio_value,
            gross_exposure=float(valuation.gross_exposure),
            net_exposure=float(valuation.net_exposure),
            volatility_daily=vol_daily,
            volatility_annualized=vol_annual,
            var_historical=var_hist,
            var_parametric=var_param,
            drawdown=DrawdownResultSchema(
                max_drawdown=dd.max_drawdown, peak_date=dd.peak_date, trough_date=dd.trough_date
            ),
            weights=valuation.weights,
            correlation_matrix={
                a: {b: float(corr.loc[a, b]) for b in corr.columns} for a in corr.index
            },
            high_correlation_pairs=[
                CorrelationPair(ticker_a=a, ticker_b=b, correlation=rho) for a, b, rho in pairs
            ],
            risk_contributions=[
                RiskContributionSchema(
                    ticker=c.ticker, weight=c.weight, marginal=c.marginal,
                    component=c.component, percent=c.percent,
                )
                for c in contrib.contributions
            ],
        )

    def _persist(
        self,
        *,
        portfolio_id: int,
        as_of: date | None,
        confidence_level: float,
        var_value: float,
        parametric_var_value: float,
        vol_annual: float,
        max_drawdown: float,
        gross: float,
        net: float,
    ) -> None:
        self.risk_metrics.add(
            RiskMetric(
                portfolio_id=portfolio_id,
                calculation_date=as_of or date.today(),
                confidence_level=Decimal(str(confidence_level)),
                var=Decimal(str(round(var_value, 6))),
                parametric_var=Decimal(str(round(parametric_var_value, 6))),
                volatility_annualized=Decimal(str(round(vol_annual, 6))),
                max_drawdown=Decimal(str(round(max_drawdown, 6))),
                exposure_gross=Decimal(str(round(gross, 6))),
                exposure_net=Decimal(str(round(net, 6))),
                method="historical",
            )
        )
        self.session.commit()


def _bars_to_frame(bars: list) -> "object":
    import pandas as pd

    return pd.DataFrame(
        {
            "date": [b.date for b in bars],
            "ticker": [b.ticker for b in bars],
            "adjusted_close": [float(b.adjusted_close) for b in bars],
        }
    )


def _as_date(idx) -> date | None:
    import pandas as pd

    if isinstance(idx, pd.Timestamp):
        return idx.date()
    return idx if isinstance(idx, date) else None
