"""Idempotently seed a fully populated demo: market data + a 'Tech Growth' portfolio.

Safe to run repeatedly (market-data upsert is idempotent; the portfolio is created
only if absent). Used by the Docker entrypoint when ``SEED_ON_START=true`` and runnable
locally with ``python -m scripts.seed_demo``.
"""

from __future__ import annotations

from decimal import Decimal

from app.db.database import get_session_factory
from app.pipelines.synthetic_data import generate_market_data
from app.schemas.portfolio import PortfolioCreate, PositionCreate
from app.services.market_data_service import MarketDataService
from app.services.portfolio_service import PortfolioService

DEMO_NAME = "Tech Growth"
DEMO_POSITIONS = [
    ("AAPL", 100, 150), ("MSFT", 80, 250), ("NVDA", 50, 200),
    ("AMZN", 60, 130), ("GOOGL", 40, 120),
]


def main() -> None:
    session = get_session_factory()()
    try:
        md = MarketDataService(session)
        if not md.repo.distinct_tickers():
            print("Loading synthetic market data...")
            summary = md.ingest_dataframe(generate_market_data())
            print(f"  ingested {summary.rows_written} rows")
        else:
            print("Market data already present; skipping load.")

        ps = PortfolioService(session)
        if ps.portfolios.get_by_name(DEMO_NAME) is None:
            print(f"Creating '{DEMO_NAME}' portfolio...")
            pf = ps.create_portfolio(PortfolioCreate(name=DEMO_NAME, description="Demo tech book"))
            for ticker, qty, avg in DEMO_POSITIONS:
                ps.add_position(
                    pf.id,
                    PositionCreate(ticker=ticker, quantity=Decimal(qty), average_price=Decimal(avg)),
                )
            print(f"  created portfolio id={pf.id} with {len(DEMO_POSITIONS)} positions")
        else:
            print(f"Portfolio '{DEMO_NAME}' already exists; skipping.")
        print("Demo seed complete.")
    finally:
        session.close()


if __name__ == "__main__":
    main()
