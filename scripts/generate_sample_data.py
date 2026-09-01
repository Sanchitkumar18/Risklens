"""CLI: generate the reproducible synthetic sample dataset to a CSV file.

Usage:
    python -m scripts.generate_sample_data
    python -m scripts.generate_sample_data --start 2020-01-01 --end 2024-12-31 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.pipelines.synthetic_data import DEFAULT_ASSETS, generate_market_data

DEFAULT_OUTPUT = Path("data/sample/market_data_sample.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic market data (demo only).")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = generate_market_data(start=args.start, end=args.end, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    tickers = sorted(df["ticker"].unique())
    per_ticker = len(df) // len(tickers)
    print("Synthetic market data generated (SYNTHETIC / demo only).")
    print(f"  rows      : {len(df):,}")
    print(f"  tickers   : {', '.join(tickers)} ({len(tickers)})")
    print(f"  per ticker: {per_ticker:,} business days")
    print(f"  range     : {df['date'].min()} → {df['date'].max()}")
    print(f"  assets    : {', '.join(a.ticker for a in DEFAULT_ASSETS)}")
    print(f"  written to: {args.output}")


if __name__ == "__main__":
    main()
