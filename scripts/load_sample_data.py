"""CLI: load a market-data CSV into the database (idempotent).

Generates the sample dataset first if the CSV is absent. Requires a reachable
database (``DATABASE_URL``) with migrations applied (``make migrate``).

Usage:
    python -m scripts.load_sample_data
    python -m scripts.load_sample_data --csv data/sample/market_data_sample.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.db.database import get_session_factory
from app.pipelines.synthetic_data import generate_market_data
from app.services.market_data_service import MarketDataService

DEFAULT_CSV = Path("data/sample/market_data_sample.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load market-data CSV into the database.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"{args.csv} not found — generating synthetic dataset first...")
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        generate_market_data().to_csv(args.csv, index=False)

    factory = get_session_factory()
    with factory() as session:
        service = MarketDataService(session)
        summary = service.ingest_csv(str(args.csv))

    print("Load complete.")
    print(f"  rows read   : {summary.rows_read:,}")
    print(f"  rows written: {summary.rows_written:,}")
    print(f"  tickers     : {', '.join(summary.tickers)}")


if __name__ == "__main__":
    main()
