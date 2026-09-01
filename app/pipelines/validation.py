"""Market-data validation stage.

Surveys a normalized raw frame *leniently* (coercing types with ``errors="coerce"``
rather than raising) so it can report **every** problem in one pass and quarantine
bad rows, instead of failing on the first. Each check is either:

* **ERROR**   – the row is rejected (cannot be trusted for analytics), or
* **WARNING** – the row is kept but flagged (suspicious but potentially legitimate,
  e.g. a genuine crash-day price jump).

The result carries an ``accepted`` frame (fed onward to cleaning → upsert), a
``rejected`` frame (with reasons), and a :class:`ValidationReport`. Thresholds are
parameters so limits are configurable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import pandas as pd

from app.core.exceptions import DataValidationError
from app.pipelines.ingestion import (
    CANONICAL_COLUMNS,
    PRICE_COLUMNS,
    normalize_columns,
)
from app.schemas.validation import ValidationIssue, ValidationReport

# ── Issue categories ────────────────────────────────────────
ERROR = "ERROR"
WARNING = "WARNING"

CAT_UNPARSEABLE_DATE = "UNPARSEABLE_DATE"
CAT_FUTURE_DATE = "FUTURE_DATE"
CAT_NULL_PRICE = "NULL_PRICE"
CAT_NON_POSITIVE_PRICE = "NON_POSITIVE_PRICE"
CAT_HIGH_LT_LOW = "HIGH_LT_LOW"
CAT_OHLC_INCONSISTENT = "OHLC_INCONSISTENT"
CAT_BAD_VOLUME = "BAD_VOLUME"
CAT_INVALID_TICKER = "INVALID_TICKER"
CAT_DUPLICATE_ROW = "DUPLICATE_ROW"
CAT_ABNORMAL_JUMP = "ABNORMAL_PRICE_JUMP"
CAT_PRICE_OUT_OF_BOUNDS = "PRICE_OUT_OF_BOUNDS"
CAT_MISSING_DATE = "MISSING_DATE"

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,15}$")
_MAX_SAMPLE_ISSUES = 20


@dataclass
class ValidationResult:
    """Bundle of accepted/rejected frames and the report."""

    accepted: pd.DataFrame
    rejected: pd.DataFrame
    report: ValidationReport


def validate_market_data(
    df: pd.DataFrame,
    *,
    max_daily_move: float = 0.5,
    min_price: float = 0.01,
    max_price: float = 1_000_000.0,
    reference_date: date | None = None,
) -> ValidationResult:
    """Validate a normalized market-data frame and split accepted/rejected rows.

    Args:
        df: raw (or normalized) market-data frame.
        max_daily_move: abnormal-jump warning threshold on |daily return|.
        min_price / max_price: sane-price warning bounds.
        reference_date: "today" for the future-date check (defaults to ``date.today()``).
    """
    reference_date = reference_date or date.today()
    work = normalize_columns(df).copy().reset_index(drop=True)

    missing = [c for c in CANONICAL_COLUMNS if c not in work.columns]
    if missing:
        raise DataValidationError(
            "Input is missing required columns.", details={"missing": missing}
        )

    n = len(work)
    ref_ts = pd.Timestamp(reference_date)

    # ── Coerce a working copy (lenient) ─────────────────────
    dt = pd.to_datetime(work["date"], errors="coerce")
    prices = {c: pd.to_numeric(work[c], errors="coerce") for c in PRICE_COLUMNS}
    volume = pd.to_numeric(work["volume"], errors="coerce")
    ticker = work["ticker"].astype(str).str.strip().str.upper()

    price_df = pd.DataFrame(prices)

    # ── ERROR masks ─────────────────────────────────────────
    m_unparse_date = dt.isna()
    m_future = dt.notna() & (dt > ref_ts)
    m_null_price = price_df.isna().any(axis=1)
    m_nonpos = (price_df <= 0).any(axis=1).fillna(False)
    m_high_low = (prices["high"] < prices["low"]).fillna(False)
    m_ohlc = (
        (prices["close"] > prices["high"])
        | (prices["close"] < prices["low"])
        | (prices["open"] > prices["high"])
        | (prices["open"] < prices["low"])
    ).fillna(False)
    m_bad_volume = volume.isna() | (volume < 0)
    m_bad_ticker = ~ticker.str.match(_TICKER_RE)

    error_masks: dict[str, pd.Series] = {
        CAT_UNPARSEABLE_DATE: m_unparse_date,
        CAT_FUTURE_DATE: m_future,
        CAT_NULL_PRICE: m_null_price,
        CAT_NON_POSITIVE_PRICE: m_nonpos,
        CAT_HIGH_LT_LOW: m_high_low,
        CAT_OHLC_INCONSISTENT: m_ohlc,
        CAT_BAD_VOLUME: m_bad_volume,
        CAT_INVALID_TICKER: m_bad_ticker,
    }

    # ── WARNING masks ───────────────────────────────────────
    m_out_bounds = dt.notna() & (
        (prices["close"] < min_price) | (prices["close"] > max_price)
    ).fillna(False)

    # Abnormal jump: |pct change of close| per ticker, chronologically.
    order = pd.DataFrame({"t": ticker, "d": dt, "c": prices["close"]})
    order_sorted = order.sort_values(["t", "d"], kind="stable")
    jump = order_sorted.groupby("t", sort=False)["c"].pct_change(fill_method=None).abs()
    m_jump = (jump.reindex(work.index) > max_daily_move).fillna(False)

    # Duplicate (ticker, date): flag the later occurrences (kept until cleaning).
    m_duplicate = pd.Series(
        pd.DataFrame({"t": ticker, "d": dt}).duplicated(keep="first").to_numpy(),
        index=work.index,
    )

    warning_masks: dict[str, pd.Series] = {
        CAT_PRICE_OUT_OF_BOUNDS: m_out_bounds,
        CAT_ABNORMAL_JUMP: m_jump,
        CAT_DUPLICATE_ROW: m_duplicate,
    }

    # ── Aggregate ───────────────────────────────────────────
    rejection = pd.Series(False, index=work.index)
    for mask in error_masks.values():
        rejection = rejection | mask.reindex(work.index).fillna(False)

    errors_by_category = {
        cat: int(mask.reindex(work.index).fillna(False).sum())
        for cat, mask in error_masks.items()
    }
    errors_by_category = {k: v for k, v in errors_by_category.items() if v > 0}

    warnings_by_category = {
        cat: int(mask.sum()) for cat, mask in warning_masks.items()
    }

    # Dataset-level: missing business days per ticker (warning, count only).
    missing_dates = _count_missing_business_days(ticker, dt)
    if missing_dates:
        warnings_by_category[CAT_MISSING_DATE] = missing_dates
    warnings_by_category = {k: v for k, v in warnings_by_category.items() if v > 0}

    sample_issues = _build_sample_issues(
        work, ticker, dt, error_masks, warning_masks
    )

    report = ValidationReport(
        rows_processed=n,
        rows_accepted=int((~rejection).sum()),
        rows_rejected=int(rejection.sum()),
        errors_by_category=errors_by_category,
        warnings_by_category=warnings_by_category,
        sample_issues=sample_issues,
    )

    accepted = work.loc[~rejection].reset_index(drop=True)
    rejected = work.loc[rejection].copy()
    if not rejected.empty:
        rejected["reject_reasons"] = _reasons_for_rows(rejected.index, error_masks, work.index)
    rejected = rejected.reset_index(drop=True)

    return ValidationResult(accepted=accepted, rejected=rejected, report=report)


def _count_missing_business_days(ticker: pd.Series, dt: pd.Series) -> int:
    """Count business days absent between each ticker's first and last date."""
    total = 0
    frame = pd.DataFrame({"t": ticker, "d": dt}).dropna(subset=["d"])
    for _, grp in frame.groupby("t", sort=False):
        present = set(grp["d"].dt.normalize())
        if not present:
            continue
        expected = pd.bdate_range(min(present), max(present))
        total += sum(1 for d in expected if d not in present)
    return total


def _reasons_for_rows(
    row_index: pd.Index, error_masks: dict[str, pd.Series], base_index: pd.Index
) -> list[str]:
    reasons: list[str] = []
    for idx in row_index:
        cats = [
            cat
            for cat, mask in error_masks.items()
            if bool(mask.reindex(base_index).fillna(False).loc[idx])
        ]
        reasons.append(",".join(cats))
    return reasons


def _build_sample_issues(
    work: pd.DataFrame,
    ticker: pd.Series,
    dt: pd.Series,
    error_masks: dict[str, pd.Series],
    warning_masks: dict[str, pd.Series],
) -> list[ValidationIssue]:
    """Collect a capped, representative sample of concrete issues."""
    issues: list[ValidationIssue] = []
    for severity, masks in ((ERROR, error_masks), (WARNING, warning_masks)):
        for cat, mask in masks.items():
            hits = list(work.index[mask.reindex(work.index).fillna(False)])[:3]
            for idx in hits:
                d = dt.iloc[idx]
                issues.append(
                    ValidationIssue(
                        category=cat,
                        severity=severity,
                        message=f"{cat} at row {idx}",
                        ticker=str(ticker.iloc[idx]),
                        date=None if pd.isna(d) else d.date().isoformat(),
                        row_index=int(idx),
                    )
                )
                if len(issues) >= _MAX_SAMPLE_ISSUES:
                    return issues
    return issues
