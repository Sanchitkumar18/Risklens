"""Return correlation analysis."""

from __future__ import annotations

import pandas as pd


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of asset return columns."""
    return returns.corr()


def high_correlation_pairs(
    corr: pd.DataFrame, threshold: float = 0.8
) -> list[tuple[str, str, float]]:
    """Return distinct asset pairs with ``|correlation| ≥ threshold``.

    Scans only the upper triangle (each unordered pair once, self-pairs excluded),
    sorted by descending absolute correlation.
    """
    pairs: list[tuple[str, str, float]] = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            rho = float(corr.loc[a, b])
            if pd.notna(rho) and abs(rho) >= threshold:
                pairs.append((a, b, rho))
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    return pairs
