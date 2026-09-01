"""Drawdown analysis.

Drawdown measures decline from the running peak of the portfolio value:

    Drawdown_t     = V_t / max_{s≤t}(V_s) − 1        (≤ 0)
    MaxDrawdown    = min_t Drawdown_t                 (the worst peak-to-trough loss)

Reported as a negative fraction (e.g. −0.32 = a 32% drawdown). The detail form also
returns the peak and trough dates — used by the assistant to explain *when* the worst
loss happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from app.core.exceptions import InsufficientHistoricalData


@dataclass
class DrawdownResult:
    """Maximum drawdown with the dates of the preceding peak and the trough."""

    max_drawdown: float          # negative fraction
    peak_date: date | None
    trough_date: date | None


def drawdown_series(values: pd.Series) -> pd.Series:
    """Return the drawdown at each point: ``V_t / running_peak_t − 1`` (≤ 0)."""
    v = pd.Series(values, dtype="float64").dropna()
    if v.empty:
        raise InsufficientHistoricalData("Empty value series for drawdown.")
    running_peak = v.cummax()
    return v / running_peak - 1.0


def max_drawdown(values: pd.Series) -> float:
    """Return the maximum drawdown as a negative fraction (0 if never below peak)."""
    dd = drawdown_series(values)
    return float(dd.min())


def max_drawdown_detail(values: pd.Series) -> DrawdownResult:
    """Return the max drawdown plus the peak and trough dates that produced it."""
    v = pd.Series(values, dtype="float64").dropna()
    if len(v) < 2:
        raise InsufficientHistoricalData(
            "Not enough observations for drawdown.", details={"observations": len(v)}
        )
    dd = v / v.cummax() - 1.0
    trough_idx = dd.idxmin()
    # The peak is the max value on/before the trough.
    peak_idx = v.loc[:trough_idx].idxmax()

    def _as_date(idx) -> date | None:
        if isinstance(idx, pd.Timestamp):
            return idx.date()
        return idx if isinstance(idx, date) else None

    return DrawdownResult(
        max_drawdown=float(dd.min()),
        peak_date=_as_date(peak_idx),
        trough_date=_as_date(trough_idx),
    )
