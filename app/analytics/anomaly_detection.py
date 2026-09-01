"""Unsupervised anomaly detection with Isolation Forest (pure).

**Why Isolation Forest?** Market anomalies are unlabeled, rare, and multivariate. An
Isolation Forest isolates points by random axis-aligned splits: anomalies lie in sparse
regions and get separated in *fewer* splits (shorter average path length → higher
anomaly score). It is (a) unsupervised — no labeled crashes needed; (b) near-linear and
scalable; (c) distribution-free — no Gaussian assumption, unlike a z-score or Mahalanobis
rule; and (d) genuinely multivariate — it flags an unusual *combination* (e.g. a modest
price move on freakish volume) that per-feature thresholds miss.

Features (engineered upstream by ``app.pipelines.transformation.enrich``):
``simple_return``, ``rolling_vol_21``, ``volume_change``, ``dist_from_ma_20``.

Detection runs **per ticker** (each asset has its own scale/behavior). Features are
standardized per ticker before fitting; ``contamination`` (expected anomaly fraction)
is configurable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES = ["simple_return", "rolling_vol_21", "volume_change", "dist_from_ma_20"]

# Map the dominant deviating feature to a human-readable anomaly type.
_FEATURE_TO_TYPE = {
    "simple_return": "price_move",
    "rolling_vol_21": "volatility_spike",
    "volume_change": "volume_spike",
    "dist_from_ma_20": "trend_deviation",
}

_OUTPUT_COLS = ["ticker", "date", "anomaly_score", "is_anomaly", "anomaly_type"]


def detect_anomalies(
    enriched: pd.DataFrame,
    feature_cols: list[str] | None = None,
    contamination: float = 0.02,
    random_state: int = 42,
    min_samples: int = 30,
) -> pd.DataFrame:
    """Flag anomalies per ticker using an Isolation Forest.

    Args:
        enriched: frame with ``ticker``, ``date``, and the feature columns.
        feature_cols: features to use (defaults to :data:`DEFAULT_FEATURES`).
        contamination: expected fraction of anomalies (Isolation Forest threshold).
        random_state: seed for reproducibility.
        min_samples: tickers with fewer usable rows are skipped (marked non-anomalous).

    Returns:
        Frame with ``ticker, date`` + the feature values + ``anomaly_score``
        (higher = more anomalous), ``is_anomaly`` (bool), ``anomaly_type``.
    """
    feature_cols = feature_cols or DEFAULT_FEATURES
    missing = [c for c in feature_cols if c not in enriched.columns]
    if missing:
        raise ValueError(f"Enriched frame missing feature columns: {missing}")

    results: list[pd.DataFrame] = []
    for ticker, grp in enriched.groupby("ticker", sort=False):
        block = grp.dropna(subset=feature_cols).copy()
        block = block.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)

        if len(block) < min_samples:
            block["anomaly_score"] = 0.0
            block["is_anomaly"] = False
            block["anomaly_type"] = None
            results.append(block)
            continue

        x_raw = block[feature_cols].to_numpy(dtype=float)
        x = StandardScaler().fit_transform(x_raw)

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
            n_jobs=1,
        )
        labels = model.fit_predict(x)  # -1 anomaly, 1 normal
        # score_samples: lower = more abnormal → negate so higher = more anomalous.
        scores = -model.score_samples(x)

        block["anomaly_score"] = scores
        block["is_anomaly"] = labels == -1
        block["anomaly_type"] = _classify(x, feature_cols, block["is_anomaly"].to_numpy())
        results.append(block)

    combined = pd.concat(results, ignore_index=True)
    keep = feature_cols + _OUTPUT_COLS
    return combined[[c for c in keep if c in combined.columns]]


def _classify(x_scaled: np.ndarray, feature_cols: list[str], is_anomaly: np.ndarray) -> list:
    """Label each anomalous row by its most-deviating standardized feature."""
    types: list = []
    dominant = np.abs(x_scaled).argmax(axis=1)
    for i, flag in enumerate(is_anomaly):
        if not flag:
            types.append(None)
        else:
            types.append(_FEATURE_TO_TYPE.get(feature_cols[dominant[i]], "multivariate"))
    return types
