"""Market-data ingestion (pure parsing layer).

Turns a raw CSV / DataFrame from *any* source (the synthetic generator now, a real
vendor export later) into normalized, typed row dicts ready for the repository's
``bulk_upsert``. This stage only handles **shape and types**; semantic validation
(price consistency, gaps, outliers) is layered on in Phase 4 between parsing and the
database write, without changing this module's contract.
"""

from __future__ import annotations

from datetime import date as date_type
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.exceptions import DataValidationError

# Canonical schema the rest of the system speaks.
CANONICAL_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adjusted_close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close", "adjusted_close"]

# Accept common header spellings from real exports and map them to canonical names.
COLUMN_ALIASES = {
    "adj close": "adjusted_close",
    "adj_close": "adjusted_close",
    "adjclose": "adjusted_close",
    "adjusted close": "adjusted_close",
    "symbol": "ticker",
    "sym": "ticker",
    "timestamp": "date",
    "trade_date": "date",
    "vol": "volume",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case, strip, and alias column names to the canonical schema."""
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower()
        renamed[col] = COLUMN_ALIASES.get(key, key)
    return df.rename(columns=renamed)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file and normalize its column names (no type coercion yet)."""
    file_path = Path(path)
    if not file_path.exists():
        raise DataValidationError(
            f"CSV file not found: {file_path}", details={"path": str(file_path)}
        )
    df = pd.read_csv(file_path)
    return normalize_columns(df)


def _require_columns(df: pd.DataFrame) -> None:
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            "Input is missing required columns.",
            details={"missing": missing, "present": list(df.columns)},
        )


def _to_decimal(value: Any, *, field: str, ctx: dict[str, Any]) -> Decimal:
    try:
        return Decimal(str(value)).quantize(Decimal("0.000001"))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise DataValidationError(
            f"Could not parse '{field}' as a decimal.", details={"value": value, **ctx}
        ) from exc


def dataframe_to_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a normalized frame into typed row dicts for the repository.

    Coerces prices to ``Decimal`` (6dp), volume to ``int``, and date to ``date``.
    Rows with unparseable dates or non-finite prices raise ``DataValidationError``
    (bad data is never silently dropped here).
    """
    df = normalize_columns(df)
    _require_columns(df)

    # Parse dates once, vectorized.
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    if parsed_dates.isna().any():
        bad = df.loc[parsed_dates.isna(), "date"].astype(str).unique().tolist()[:5]
        raise DataValidationError("Unparseable date value(s).", details={"examples": bad})

    rows: list[dict[str, Any]] = []
    for i, record in enumerate(df.to_dict(orient="records")):
        ticker = str(record["ticker"]).strip().upper()
        row_ctx = {"row": i, "ticker": ticker}
        record_date: date_type = parsed_dates.iloc[i].date()

        row: dict[str, Any] = {"ticker": ticker, "date": record_date}
        for col in PRICE_COLUMNS:
            row[col] = _to_decimal(record[col], field=col, ctx=row_ctx)
        try:
            row["volume"] = int(record["volume"])
        except (ValueError, TypeError) as exc:
            raise DataValidationError(
                "Could not parse 'volume' as an integer.",
                details={"value": record["volume"], **row_ctx},
            ) from exc
        rows.append(row)
    return rows
