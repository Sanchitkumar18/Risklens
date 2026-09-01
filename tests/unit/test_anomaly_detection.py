"""Unit tests for the Isolation Forest anomaly detector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analytics.anomaly_detection import DEFAULT_FEATURES, detect_anomalies


def _features(n: int = 250, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ticker": ["AAPL"] * n,
            "date": pd.bdate_range("2022-01-03", periods=n),
            "simple_return": rng.normal(0, 0.02, n),
            "rolling_vol_21": np.abs(rng.normal(0.02, 0.004, n)),
            "volume_change": rng.normal(0, 0.3, n),
            "dist_from_ma_20": rng.normal(0, 0.05, n),
        }
    )


@pytest.mark.unit
def test_detects_injected_return_spike():
    df = _features()
    spike_date = df.loc[120, "date"]
    df.loc[120, "simple_return"] = -0.35  # ~17 sigma crash
    out = detect_anomalies(df, contamination=0.03)
    row = out[out["date"] == spike_date].iloc[0]
    assert bool(row["is_anomaly"]) is True
    assert row["anomaly_type"] == "price_move"


@pytest.mark.unit
def test_detects_injected_volume_spike():
    df = _features(seed=1)
    vdate = df.loc[60, "date"]
    df.loc[60, "volume_change"] = 9.0  # enormous volume surge
    out = detect_anomalies(df, contamination=0.03)
    row = out[out["date"] == vdate].iloc[0]
    assert bool(row["is_anomaly"]) is True
    assert row["anomaly_type"] == "volume_spike"


@pytest.mark.unit
def test_reproducible():
    df = _features()
    a = detect_anomalies(df, contamination=0.03, random_state=7)
    b = detect_anomalies(df, contamination=0.03, random_state=7)
    assert a["is_anomaly"].tolist() == b["is_anomaly"].tolist()


@pytest.mark.unit
def test_contamination_controls_count():
    df = _features(seed=2)
    low = detect_anomalies(df, contamination=0.02)["is_anomaly"].sum()
    high = detect_anomalies(df, contamination=0.10)["is_anomaly"].sum()
    assert high > low


@pytest.mark.unit
def test_too_few_samples_flags_nothing():
    df = _features(n=15)
    out = detect_anomalies(df, min_samples=30)
    assert out["is_anomaly"].sum() == 0


@pytest.mark.unit
def test_missing_feature_column_raises():
    df = _features().drop(columns=["volume_change"])
    with pytest.raises(ValueError):
        detect_anomalies(df, feature_cols=DEFAULT_FEATURES)
