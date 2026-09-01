"""Unit tests for correlation analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.risk.correlation import correlation_matrix, high_correlation_pairs


@pytest.mark.unit
def test_perfect_and_anti_correlation():
    base = pd.Series(np.linspace(-0.05, 0.05, 50))
    df = pd.DataFrame({"A": base, "B": base * 2.0, "C": -base})
    corr = correlation_matrix(df)
    assert corr.loc["A", "B"] == pytest.approx(1.0)
    assert corr.loc["A", "C"] == pytest.approx(-1.0)


@pytest.mark.unit
def test_high_correlation_pairs_threshold():
    base = pd.Series(np.linspace(-0.05, 0.05, 50))
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"A": base, "B": base * 2.0, "D": pd.Series(rng.normal(size=50))})
    pairs = high_correlation_pairs(correlation_matrix(df), threshold=0.8)
    names = {frozenset((a, b)) for a, b, _ in pairs}
    assert frozenset(("A", "B")) in names
    # A/D and B/D are (near) uncorrelated and should not appear.
    assert all("D" not in p for p in names)


@pytest.mark.unit
def test_single_asset_matrix():
    df = pd.DataFrame({"A": [0.01, -0.01, 0.02, 0.0]})
    corr = correlation_matrix(df)
    assert corr.shape == (1, 1)
    assert high_correlation_pairs(corr) == []
