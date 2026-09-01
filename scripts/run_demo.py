"""End-to-end demo runner.

Runs the entire RiskLens workflow in one process and prints each step — data load,
portfolio construction, valuation, risk, anomalies, alerts, stress, and grounded
assistant answers. Self-contained: defaults to a local SQLite file so it needs no
Postgres.

    python -m scripts.run_demo         (or: make demo)
"""

from __future__ import annotations

import os

# Default to a local SQLite DB before any app import reads settings.
os.environ.setdefault("DATABASE_URL", "sqlite:///risklens_demo.db")
os.environ.setdefault("LOG_LEVEL", "WARNING")

from decimal import Decimal  # noqa: E402

from app.db.database import Base, get_engine, get_session_factory  # noqa: E402
from app.pipelines.synthetic_data import generate_market_data  # noqa: E402
from app.schemas.assistant import AssistantQuery  # noqa: E402
from app.schemas.portfolio import PortfolioCreate, PositionCreate  # noqa: E402
from app.services.alert_service import AlertService  # noqa: E402
from app.services.anomaly_service import AnomalyService  # noqa: E402
from app.services.assistant_service import AssistantService  # noqa: E402
from app.services.market_data_service import MarketDataService  # noqa: E402
from app.services.portfolio_service import PortfolioService  # noqa: E402
from app.services.risk_service import RiskService  # noqa: E402
from app.services.stress_service import StressService  # noqa: E402

POSITIONS = [("AAPL", 100, 150), ("MSFT", 80, 250), ("NVDA", 50, 200), ("AMZN", 60, 130), ("GOOGL", 40, 120)]
QUESTIONS = [
    "Why is my portfolio risky right now?",
    "Which asset contributes most to portfolio risk?",
    "Explain my 99% VaR in simple terms.",
    "What happens if technology stocks fall 25%?",
    "What are my current risk alerts?",
]


def _h(title: str) -> None:
    print("\n" + "═" * 74 + f"\n  {title}\n" + "═" * 74)


def main() -> None:
    import app.db.models  # noqa: F401  register tables

    Base.metadata.create_all(get_engine())
    session = get_session_factory()()

    try:
        _h("1. Load synthetic market data (validated pipeline)")
        summary = MarketDataService(session).ingest_dataframe(generate_market_data())
        print(f"  rows written: {summary.rows_written:,} | rejected: {summary.rows_rejected} | tickers: {', '.join(summary.tickers)}")

        _h("2. Build the 'Tech Growth' portfolio")
        ps = PortfolioService(session)
        existing = ps.portfolios.get_by_name("Tech Growth")
        if existing:
            pid = existing.id
            print("  portfolio already exists; reusing it")
        else:
            pf = ps.create_portfolio(PortfolioCreate(name="Tech Growth", description="Demo tech book"))
            pid = pf.id
            for t, q, ap in POSITIONS:
                ps.add_position(pid, PositionCreate(ticker=t, quantity=Decimal(q), average_price=Decimal(ap)))
            print("  positions: " + ", ".join(f"{t} {q}@${ap}" for t, q, ap in POSITIONS))

        _h("3. Valuation")
        v = ps.value_portfolio(pid)
        print(f"  total value: ${float(v.total_value):,.0f} | unrealized P&L: ${float(v.unrealized_pnl):,.0f}")
        for h in v.holdings:
            print(f"    {h.ticker:5} weight {h.weight:6.1%}  market value ${float(h.market_value):,.0f}")

        _h("4. Risk metrics (99% confidence)")
        report = RiskService(session).compute_metrics(pid, confidence_level=0.99)
        print(f"  annualized volatility: {report.volatility_annualized:.1%}")
        print(f"  max drawdown: {report.drawdown.max_drawdown:.1%} ({report.drawdown.peak_date} → {report.drawdown.trough_date})")
        for v_ in report.var_historical:
            print(f"    {v_.confidence_level:.0%} 1-day VaR: {v_.var_fraction:.2%}  (${v_.var_value:,.0f})")
        print("  risk contribution:")
        for c in report.risk_contributions:
            print(f"    {c.ticker:5} weight {c.weight:6.1%} → risk {c.percent:6.1%}")

        _h("5. Anomaly detection (Isolation Forest)")
        an = AnomalyService(session).scan_portfolio(pid, contamination=0.02)
        print(f"  analyzed {an.rows_analyzed} rows, found {an.anomalies_found} anomalies")
        for a in an.anomalies[:3]:
            print(f"    {a.date} {a.ticker:5} {a.anomaly_type:16} score {a.anomaly_score:.3f}")

        _h("6. Risk alerts")
        al = AlertService(session).evaluate(pid)
        print(f"  {al.breaches} breach(es) by severity: {al.by_severity}")
        for a in sorted(al.alerts, key=lambda x: x.severity)[:6]:
            print(f"    [{a.severity:8}] {a.alert_type}: {a.message}")

        _h("7. Stress test — 25% technology selloff")
        st = StressService(session).run_builtin(pid, "tech_selloff")
        print(f"  loss ${float(st.total_loss):,.0f} ({st.pct_loss:.1%}) | worst: {', '.join(st.worst_assets)}")

        _h("8. GenAI assistant (grounded answers)")
        asst = AssistantService(session)
        for q in QUESTIONS:
            resp = asst.query(AssistantQuery(portfolio_id=pid, question=q, confidence=0.99 if "99" in q else None))
            print(f"\n  Q: {q}\n  A: {resp.answer}")

        print("\n" + "═" * 74)
        print("  Demo complete. Start the API (`make run`) + dashboard (`make dashboard`)")
        print("  to explore interactively, or `docker compose up --build`.")
        print("═" * 74)
    finally:
        session.close()


if __name__ == "__main__":
    main()
