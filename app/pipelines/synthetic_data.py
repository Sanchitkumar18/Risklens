"""Synthetic (demo) market-data generator.

**This produces SYNTHETIC data for local development and demos only — it is not real
market data.** It is fully reproducible (seeded) so results are stable across runs.

Model
-----
Returns are simulated with a **multi-factor model** (finance-credible and guaranteed to
produce a positive-semi-definite correlation structure, unlike hand-tuning a raw
correlation matrix)::

    r_{i,t} = mu_i/252  +  beta^mkt_i · f^mkt_t  +  beta^tech_i · f^tech_t  +  e_i,t

* ``f^mkt`` (broad market) and ``f^tech`` (technology sector) are common factors, so
  assets loading on the same factor become correlated — SPY tracks the market factor,
  tech names load on both.
* Each factor and each idiosyncratic series ``e_i`` follows a **GARCH(1,1)** variance
  process, which reproduces *volatility clustering* (calm and turbulent regimes).
* Rare **market crash days** inject a large negative common shock (fat left tail);
  rare **idiosyncratic jumps** inject single-name anomalies for the detector to find.

OHLC bars are derived from the simulated close with realistic intraday ranges and
overnight gaps; volume is driven by the day's absolute return (busy days trade more).
``adjusted_close`` equals ``close`` here — no dividends/splits are modelled (documented
limitation), and the risk engine uses ``adjusted_close`` throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ── GARCH(1,1) persistence parameters (shared shape, per-series scale) ───────
# alpha + beta < 1 for stationarity; ~0.98 gives realistic, persistent clustering.
_GARCH_ALPHA = 0.08
_GARCH_BETA = 0.90


@dataclass(frozen=True)
class AssetSpec:
    """Static parameters describing one synthetic asset."""

    ticker: str
    name: str
    sector: str  # "broad" | "tech"
    init_price: float
    annual_drift: float
    idio_vol_daily: float  # unconditional daily idiosyncratic vol
    beta_market: float
    beta_tech: float
    base_volume: int


# Seven assets: a broad-market ETF plus six large-cap tech names. Betas/vols are
# chosen so SPY is the least volatile and NVDA/TSLA the most — asserted in tests.
DEFAULT_ASSETS: list[AssetSpec] = [
    AssetSpec("SPY", "SPDR S&P 500 ETF", "broad", 400.0, 0.09, 0.0030, 1.00, 0.00, 70_000_000),
    AssetSpec("AAPL", "Apple Inc.", "tech", 150.0, 0.15, 0.0080, 1.05, 0.45, 90_000_000),
    AssetSpec("MSFT", "Microsoft Corp.", "tech", 250.0, 0.16, 0.0075, 1.00, 0.50, 30_000_000),
    AssetSpec("GOOGL", "Alphabet Inc.", "tech", 120.0, 0.13, 0.0090, 1.05, 0.45, 28_000_000),
    AssetSpec("AMZN", "Amazon.com Inc.", "tech", 130.0, 0.14, 0.0110, 1.10, 0.40, 45_000_000),
    AssetSpec("NVDA", "NVIDIA Corp.", "tech", 200.0, 0.35, 0.0180, 1.25, 0.65, 50_000_000),
    AssetSpec("TSLA", "Tesla Inc.", "tech", 220.0, 0.22, 0.0220, 1.20, 0.35, 110_000_000),
]

# ── Factor unconditional daily volatilities ─────────────────────────────────
_MARKET_VOL_DAILY = 0.0090   # ~14% annualized
_TECH_VOL_DAILY = 0.0070     # ~11% annualized sector factor

# Column order for the emitted long-format frame.
OUTPUT_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adjusted_close", "volume"]


@dataclass
class _GarchConfig:
    sigma_bar: np.ndarray  # unconditional daily vol per series
    alpha: float = _GARCH_ALPHA
    beta: float = _GARCH_BETA
    shocks: np.ndarray = field(default=None)  # standardized innovations (T x S)


def _simulate_garch(cfg: _GarchConfig) -> np.ndarray:
    """Run a GARCH(1,1) recursion for several independent series at once.

    Returns an ``(T, S)`` array of innovations ``e_{t} = sigma_{t} · z_{t}`` where the
    conditional variance evolves as ``sigma^2_t = omega + alpha·e^2_{t-1} + beta·sigma^2_{t-1}``.
    The time loop is inherent to GARCH; assets are vectorized within each step.
    """
    z = cfg.shocks
    T, S = z.shape
    omega = cfg.sigma_bar**2 * (1.0 - cfg.alpha - cfg.beta)

    sigma2 = np.empty((T, S))
    e = np.empty((T, S))
    sigma2[0] = cfg.sigma_bar**2
    e[0] = np.sqrt(sigma2[0]) * z[0]
    for t in range(1, T):
        sigma2[t] = omega + cfg.alpha * e[t - 1] ** 2 + cfg.beta * sigma2[t - 1]
        e[t] = np.sqrt(sigma2[t]) * z[t]
    return e


def _build_shocks(
    rng: np.random.Generator,
    n_days: int,
    n_assets: int,
    n_crashes: int,
    n_idio_jumps: int,
) -> np.ndarray:
    """Standardized innovations for [market, tech, idio_1..idio_n], with tail events.

    Crash days push the *market* innovation deep negative (a correlated sell-off);
    idiosyncratic jumps push a single asset's innovation to an extreme (an anomaly).
    """
    n_series = 2 + n_assets
    z = rng.standard_normal((n_days, n_series))

    # Candidate days for tail events, buffered from the edges. The buffer adapts to
    # short series so small date ranges still work; long ranges keep the original
    # 30/5 buffer (so the default dataset is unchanged).
    lo = 30 if n_days > 60 else max(1, n_days // 10)
    hi = n_days - (5 if n_days > 60 else 1)
    population = np.arange(lo, hi)
    if population.size == 0:
        return z

    # Market crash days: large negative common shock.
    n_crashes = min(n_crashes, population.size)
    if n_crashes > 0:
        crash_days = rng.choice(population, size=n_crashes, replace=False)
        z[crash_days, 0] = rng.uniform(-6.0, -3.5, size=n_crashes)

    # Idiosyncratic single-name jumps (both signs) → labelled anomalies to detect.
    n_idio_jumps = min(n_idio_jumps, population.size)
    if n_idio_jumps > 0:
        jump_days = rng.choice(population, size=n_idio_jumps, replace=False)
        jump_assets = rng.integers(0, n_assets, size=n_idio_jumps)
        signs = rng.choice([-1.0, 1.0], size=n_idio_jumps)
        z[jump_days, 2 + jump_assets] = signs * rng.uniform(4.5, 7.5, size=n_idio_jumps)

    return z


def _derive_ohlcv(
    rng: np.random.Generator,
    close: np.ndarray,
    log_returns: np.ndarray,
    init_price: float,
    base_volume: int,
) -> dict[str, np.ndarray]:
    """Build open/high/low/volume around a close path (all positive, high≥low)."""
    n = close.shape[0]
    daily_sigma = float(np.std(log_returns)) or 0.01

    prev_close = np.empty(n)
    prev_close[0] = init_price
    prev_close[1:] = close[:-1]

    # Overnight gap → open.
    gap = rng.normal(0.0, 0.35 * daily_sigma, size=n)
    open_ = prev_close * np.exp(gap)

    top = np.maximum(open_, close)
    bottom = np.minimum(open_, close)
    high = top * np.exp(np.abs(rng.normal(0.0, 0.45 * daily_sigma, size=n)))
    low = bottom * np.exp(-np.abs(rng.normal(0.0, 0.45 * daily_sigma, size=n)))

    # Volume scales with the day's absolute return plus lognormal noise.
    abs_ret = np.abs(log_returns)
    std_abs = (abs_ret - abs_ret.mean()) / (abs_ret.std() or 1.0)
    volume = base_volume * np.exp(0.45 * std_abs + rng.normal(0.0, 0.30, size=n))
    volume = np.maximum(volume.astype(np.int64), 1)

    return {"open": open_, "high": high, "low": low, "volume": volume}


def generate_market_data(
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    assets: list[AssetSpec] | None = None,
    seed: int = 42,
    n_crashes: int = 3,
    n_idio_jumps: int = 6,
) -> pd.DataFrame:
    """Generate a reproducible synthetic OHLCV dataset in long format.

    Args:
        start, end: inclusive business-day date range.
        assets: asset specifications (defaults to :data:`DEFAULT_ASSETS`).
        seed: RNG seed for full reproducibility.
        n_crashes: number of correlated market-crash days to inject.
        n_idio_jumps: number of single-name jump anomalies to inject.

    Returns:
        DataFrame with columns :data:`OUTPUT_COLUMNS`, sorted by (ticker, date).
    """
    assets = assets or DEFAULT_ASSETS
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(start=start, end=end)
    n_days = len(dates)
    n_assets = len(assets)
    if n_days < 2:
        raise ValueError("date range must span at least two business days")

    # ── Common + idiosyncratic innovations with GARCH clustering ────────────
    z = _build_shocks(rng, n_days, n_assets, n_crashes, n_idio_jumps)
    sigma_bar = np.array(
        [_MARKET_VOL_DAILY, _TECH_VOL_DAILY] + [a.idio_vol_daily for a in assets]
    )
    e = _simulate_garch(_GarchConfig(sigma_bar=sigma_bar, shocks=z))
    f_market = e[:, 0]
    f_tech = e[:, 1]

    frames: list[pd.DataFrame] = []
    for k, spec in enumerate(assets):
        idio = e[:, 2 + k]
        log_ret = (
            spec.annual_drift / 252.0
            + spec.beta_market * f_market
            + spec.beta_tech * f_tech
            + idio
        )
        close = spec.init_price * np.exp(np.cumsum(log_ret))
        ohlcv = _derive_ohlcv(rng, close, log_ret, spec.init_price, spec.base_volume)

        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": spec.ticker,
                    "open": np.round(ohlcv["open"], 6),
                    "high": np.round(ohlcv["high"], 6),
                    "low": np.round(ohlcv["low"], 6),
                    "close": np.round(close, 6),
                    "adjusted_close": np.round(close, 6),
                    "volume": ohlcv["volume"],
                }
            )
        )

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    return result[OUTPUT_COLUMNS]
