"""Market-data cleaning stage.

Runs on rows that already passed validation. Cleaning is **explicit and reported**,
never silent: duplicates removed and gaps filled are counted and returned. By default
missing calendar days are *not* fabricated (that would invent prices) — the optional
``fill_missing_business_days`` is available but off by default and clearly flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.pipelines.ingestion import normalize_columns


@dataclass
class CleaningStats:
    """What the cleaning stage changed."""

    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    rows_filled: int = 0
    notes: list[str] = field(default_factory=list)


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop duplicate (ticker, date) rows, keeping the last occurrence.

    Keeping *last* means a corrected re-statement of a bar wins over the original.
    Returns the deduplicated frame and the count removed.
    """
    before = len(df)
    out = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    return out.reset_index(drop=True), before - len(out)


def sort_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Return bars sorted by (ticker, date) — the canonical order for analytics."""
    return df.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)


def clean_market_data(
    df: pd.DataFrame, *, fill_missing_business_days: bool = False
) -> tuple[pd.DataFrame, CleaningStats]:
    """Deduplicate, sort, and (optionally) forward-fill missing business days.

    Args:
        df: validated market-data frame.
        fill_missing_business_days: when True, reindex each ticker to a complete
            business-day calendar and forward-fill prices (volume set to 0 on filled
            days). Off by default so no synthetic prices are introduced implicitly.
    """
    work = normalize_columns(df).copy()
    work["date"] = pd.to_datetime(work["date"])
    stats = CleaningStats(rows_in=len(work))

    work, dupes = deduplicate(work)
    stats.duplicates_removed = dupes
    if dupes:
        stats.notes.append(f"Removed {dupes} duplicate (ticker, date) row(s), kept last.")

    if fill_missing_business_days:
        work, filled = _forward_fill_gaps(work)
        stats.rows_filled = filled
        if filled:
            stats.notes.append(
                f"Forward-filled {filled} missing business day(s) (volume=0 on fills)."
            )

    work = sort_bars(work)
    stats.rows_out = len(work)
    return work, stats


def _forward_fill_gaps(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    price_cols = ["open", "high", "low", "close", "adjusted_close"]
    filled_total = 0
    frames: list[pd.DataFrame] = []
    for ticker, grp in df.groupby("ticker", sort=False):
        grp = grp.sort_values("date").set_index("date")
        full = pd.bdate_range(grp.index.min(), grp.index.max())
        reindexed = grp.reindex(full)
        added = int(reindexed[price_cols[0]].isna().sum())
        filled_total += added
        reindexed[price_cols] = reindexed[price_cols].ffill()
        reindexed["volume"] = reindexed["volume"].fillna(0).astype("int64")
        reindexed["ticker"] = ticker
        reindexed = reindexed.reset_index(names="date")
        frames.append(reindexed)
    out = pd.concat(frames, ignore_index=True)
    return out, filled_total
