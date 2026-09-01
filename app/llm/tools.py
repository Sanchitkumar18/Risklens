"""Grounded analytical tools for the GenAI assistant.

**This is the anti-hallucination boundary.** Every number the assistant can ever state
must come through one of these tools, which are thin wrappers over the *same services*
the REST API uses. The LLM never computes metrics — it selects tools, reads their
structured output, and explains it. If a metric isn't returned by a tool, the assistant
cannot report it.

``RiskLensToolkit`` binds a database session and a portfolio id, so tool signatures stay
minimal (the LLM only supplies parameters like ``scenario`` or ``confidence``).
``build_langchain_tools`` exposes them as LangChain ``StructuredTool`` objects for the
LangGraph agent (Phase 13); the plain methods are also callable directly for the
deterministic mock path.
"""

from __future__ import annotations

import functools
import json
from typing import Any

from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.services.alert_service import AlertService
from app.services.anomaly_service import AnomalyService
from app.services.portfolio_service import PortfolioService
from app.services.risk_service import RiskService
from app.services.stress_service import StressService

logger = get_logger("risklens.tools")


def _dumps(obj: Any) -> str:
    """JSON-serialize tool output (handles Decimal/date via ``str``)."""
    return json.dumps(obj, default=str)


class RiskLensToolkit:
    """Portfolio-scoped grounded tools backed by the analytics services."""

    def __init__(self, session: Session, portfolio_id: int) -> None:
        self.session = session
        self.portfolio_id = portfolio_id
        self.portfolios = PortfolioService(session)
        self.risk = RiskService(session)
        self.stress = StressService(session)
        self.anomalies = AnomalyService(session)
        self.alerts = AlertService(session)

    # ── tools (each returns a JSON-serializable dict) ───────
    def get_portfolio_summary(self) -> dict[str, Any]:
        """Holdings, weights, market value, exposure, and unrealized P&L."""
        v = self.portfolios.value_portfolio(self.portfolio_id)
        return {
            "name": v.name,
            "as_of_date": str(v.as_of_date),
            "total_value": float(v.total_value),
            "gross_exposure": float(v.gross_exposure),
            "net_exposure": float(v.net_exposure),
            "unrealized_pnl": float(v.unrealized_pnl),
            "num_positions": len(v.holdings),
            "holdings": [
                {"ticker": h.ticker, "weight": h.weight, "market_value": float(h.market_value)}
                for h in v.holdings
            ],
        }

    def get_risk_metrics(self, confidence: float = 0.95) -> dict[str, Any]:
        """VaR (historical + parametric), volatility, and observation count."""
        r = self.risk.compute_metrics(self.portfolio_id, confidence_level=confidence, persist=False)
        return {
            "as_of_date": str(r.as_of_date),
            "observations": r.observations,
            "confidence_level": r.confidence_level,
            "portfolio_value": r.portfolio_value,
            "volatility_daily": r.volatility_daily,
            "volatility_annualized": r.volatility_annualized,
            "var_historical": [
                {"confidence_level": v.confidence_level, "var_fraction": v.var_fraction, "var_value": v.var_value}
                for v in r.var_historical
            ],
            "var_parametric": (
                {"confidence_level": r.var_parametric.confidence_level,
                 "var_fraction": r.var_parametric.var_fraction, "var_value": r.var_parametric.var_value}
                if r.var_parametric else None
            ),
        }

    def get_asset_exposure(self) -> dict[str, Any]:
        """Per-asset weights and gross/net exposure, with the largest position."""
        v = self.portfolios.value_portfolio(self.portfolio_id)
        largest = max(v.holdings, key=lambda h: abs(h.weight), default=None)
        return {
            "gross_exposure": float(v.gross_exposure),
            "net_exposure": float(v.net_exposure),
            "weights": v.weights,
            "largest_position": (
                {"ticker": largest.ticker, "weight": largest.weight} if largest else None
            ),
        }

    def get_correlation_matrix(self) -> dict[str, Any]:
        """Asset return correlation matrix and highly correlated pairs."""
        c = self.risk.compute_correlation(self.portfolio_id)
        return {
            "tickers": c.tickers,
            "matrix": c.matrix,
            "high_correlation_pairs": [
                {"ticker_a": p.ticker_a, "ticker_b": p.ticker_b, "correlation": p.correlation}
                for p in c.high_correlation_pairs
            ],
        }

    def get_drawdown_analysis(self) -> dict[str, Any]:
        """Maximum drawdown and the peak/trough dates that produced it."""
        r = self.risk.compute_metrics(self.portfolio_id, persist=False)
        return {
            "max_drawdown": r.drawdown.max_drawdown,
            "peak_date": str(r.drawdown.peak_date),
            "trough_date": str(r.drawdown.trough_date),
        }

    def get_risk_contributions(self) -> dict[str, Any]:
        """Each asset's share of portfolio volatility (Euler decomposition)."""
        r = self.risk.compute_metrics(self.portfolio_id, persist=False)
        contribs = [
            {"ticker": c.ticker, "weight": c.weight, "percent": c.percent}
            for c in r.risk_contributions
        ]
        top = max(contribs, key=lambda c: c["percent"], default=None)
        return {"contributions": contribs, "top_contributor": top}

    def run_stress_test(self, scenario: str = "market_crash") -> dict[str, Any]:
        """Portfolio P&L under a named scenario (market_crash, severe_crash, tech_selloff, volatility_shock)."""
        s = self.stress.run_builtin(self.portfolio_id, scenario)
        return {
            "scenario": s.scenario_name,
            "description": s.description,
            "pct_loss": s.pct_loss,
            "total_loss": float(s.total_loss),
            "worst_assets": s.worst_assets,
            "legs": [
                {"ticker": leg.ticker, "shock": leg.shock, "pnl": float(leg.pnl)} for leg in s.legs
            ],
        }

    def get_anomalies(self) -> dict[str, Any]:
        """Recently detected anomalies for the portfolio's assets (scans if none stored)."""
        stored = self.anomalies.list_persisted(self.portfolio_id, limit=10)
        if not stored:
            self.anomalies.scan_portfolio(self.portfolio_id)
            stored = self.anomalies.list_persisted(self.portfolio_id, limit=10)
        return {
            "count": len(stored),
            "anomalies": [
                {"ticker": a.ticker, "date": str(a.date), "type": a.anomaly_type, "score": a.anomaly_score}
                for a in stored
            ],
        }

    def get_alerts(self) -> dict[str, Any]:
        """Current risk alerts (evaluates thresholds if none exist yet)."""
        alerts = self.alerts.list_alerts(self.portfolio_id)
        if not alerts:
            self.alerts.evaluate(self.portfolio_id)
            alerts = self.alerts.list_alerts(self.portfolio_id)
        return {
            "count": len(alerts),
            "alerts": [
                {"type": a.alert_type, "severity": a.severity, "message": a.message,
                 "acknowledged": a.acknowledged}
                for a in alerts
            ],
        }

    # ── mapping for direct (deterministic) invocation ───────
    def as_dict(self) -> dict[str, Any]:
        """Name → bound method map, for the deterministic mock path."""
        return {
            "get_portfolio_summary": self.get_portfolio_summary,
            "get_risk_metrics": self.get_risk_metrics,
            "get_asset_exposure": self.get_asset_exposure,
            "get_correlation_matrix": self.get_correlation_matrix,
            "get_drawdown_analysis": self.get_drawdown_analysis,
            "get_risk_contributions": self.get_risk_contributions,
            "run_stress_test": self.run_stress_test,
            "get_anomalies": self.get_anomalies,
            "get_alerts": self.get_alerts,
        }


# ── LangChain tool descriptions (guide LLM tool selection) ──
_DESCRIPTIONS = {
    "get_portfolio_summary": "Get holdings, weights, market value, exposure and unrealized P&L for the portfolio.",
    "get_risk_metrics": "Get Value at Risk (historical & parametric) and volatility. Optional 'confidence' (0.90/0.95/0.99).",
    "get_asset_exposure": "Get per-asset weights, gross/net exposure and the largest single position.",
    "get_correlation_matrix": "Get the asset return correlation matrix and highly correlated pairs.",
    "get_drawdown_analysis": "Get the maximum drawdown and the peak/trough dates that produced it.",
    "get_risk_contributions": "Get each asset's percentage contribution to portfolio volatility (which asset drives risk).",
    "run_stress_test": "Run a stress scenario ('market_crash','severe_crash','tech_selloff','volatility_shock') and get the P&L.",
    "get_anomalies": "Get recently detected unusual market movements for the portfolio's assets.",
    "get_alerts": "Get the current risk-limit alerts for the portfolio.",
}


def build_langchain_tools(toolkit: RiskLensToolkit) -> list[StructuredTool]:
    """Wrap the toolkit's methods as LangChain ``StructuredTool`` objects.

    Each tool returns a JSON string (LangChain tool contract). ``portfolio_id`` is
    already bound in the toolkit, so the LLM only supplies real parameters.
    """
    methods = toolkit.as_dict()
    tools: list[StructuredTool] = []
    for name, method in methods.items():

        def _make(fn):
            # functools.wraps preserves the bound method's signature so the LLM sees
            # the real parameters (e.g. `confidence`, `scenario`), while the wrapper
            # serializes the dict result to the JSON string LangChain tools return.
            @functools.wraps(fn)
            def _tool(*args: Any, **kwargs: Any) -> str:
                logger.info("tool invoked", extra={"tool": name})
                return _dumps(fn(*args, **kwargs))

            return _tool

        tools.append(
            StructuredTool.from_function(
                func=_make(method),
                name=name,
                description=_DESCRIPTIONS[name],
            )
        )
    return tools
