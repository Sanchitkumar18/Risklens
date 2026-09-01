# RiskLens — Risk Methodology

This document defines every risk metric RiskLens computes: its mathematical
definition, formula, assumptions, implementation, a worked example, and the unit test
that pins it. All calculations live as **pure functions** in `app/risk/` and
`app/analytics/` (no I/O), orchestrated by `app/services/risk_service.py`.

> **Disclaimer:** RiskLens runs on synthetic/demo data for research purposes. VaR is
> not a maximum-loss guarantee. Nothing here is financial advice.

---

## Global conventions

| Choice | Decision | Rationale |
|---|---|---|
| Return type | **Simple** daily returns `r_t = P_t/P_{t-1} − 1` from **adjusted close** | Adjusted close handles dividends/splits; simple returns aggregate across a portfolio in one period. |
| Portfolio series | **Current holdings × historical prices**: `V_t = Σ_i q_i · P_{i,t}` | Answers "how risky is the book I hold today?" — the standard risk convention. |
| VaR sign | **Positive loss** (fraction and dollars) | Removes the #1 source of VaR ambiguity; applied consistently everywhere. |
| Volatility | Sample std, `ddof=1`; annualize `× √252` | Unbiased estimator; square-root-of-time rule. |
| Trading days/year | 252 | Configurable via `TRADING_DAYS_PER_YEAR`. |

---

## 1. Historical VaR (primary)

**Definition.** The loss threshold exceeded with probability `1 − c` over one day,
estimated directly from the empirical return distribution (no distributional assumption).

**Formula.**
```
VaR_c = − Quantile_{1−c}( r^P )          (fraction)
VaR_c$ = VaR_c · V_today                  (dollars)
```

**Assumptions.** The historical sample is representative of near-future risk;
observations are i.i.d.-ish; enough data exists to estimate the tail quantile
(RiskLens requires ≥ 20 observations).

**Implementation.** `app/risk/var.py::historical_var` (uses `np.quantile`).

**Worked example (demo portfolio, seed=42).** 99% 1-day VaR ≈ **3.09% → $5,715** on a
$184,657 book; monotone across levels (90% $2,956 ≤ 95% $3,731 ≤ 99% $5,715).

**Tests.** `tests/unit/test_var.py` — equals `−quantile`, positivity, monotonicity in
confidence, insufficient-data + bad-confidence guards.

**Interpretation to give users.** "Based on the historical return distribution, a
one-day loss worse than ~$5,715 occurred in about 1% of days." *Not* a maximum.

---

## 2. Parametric (variance–covariance) VaR

**Definition.** VaR under a Normal-returns assumption, for comparison/teaching.

**Formula.** `VaR_c = −(μ + z_{1−c}·σ)`, where `z_{1−c} = Φ⁻¹(1−c)` (negative).

**Assumptions.** Returns are Normal. This **understates tail risk** when returns are
fat-tailed — visible in the demo, where parametric 99% ($5,585) sits *below* historical
99% ($5,715).

**Implementation.** `app/risk/var.py::parametric_var` (`scipy.stats.norm`). `zero_mean`
flag drops the drift term (common for short horizons).

**Tests.** `tests/unit/test_var.py::test_parametric_var_formula` checks
`VaR = Φ⁻¹(c)·σ` exactly when `zero_mean=True`.

---

## 3. Volatility

**Definition.** Dispersion of daily returns; annualized to a yearly figure.

**Formula.** `σ_daily = std(r, ddof=1)`, `σ_annual = σ_daily · √252`.

**Assumptions.** Square-root-of-time scaling assumes returns are serially uncorrelated.

**Implementation.** `app/risk/volatility.py`.

**Worked example.** Demo portfolio ≈ **21.2%** annualized.

**Tests.** `tests/unit/test_volatility.py` — known value `0.01581…`, √252 scaling,
constant-returns → 0, insufficient-data guard.

---

## 4. Maximum Drawdown

**Definition.** The largest peak-to-trough decline in portfolio value.

**Formula.** `Drawdown_t = V_t / max_{s≤t} V_s − 1`; `MaxDrawdown = min_t Drawdown_t`
(a negative fraction).

**Implementation.** `app/risk/drawdown.py` (vectorized `cummax`); the detail form also
returns the peak and trough **dates**.

**Worked example.** Demo portfolio **−37.3%** (peak 2022-10-05 → trough 2024-01-09).

**Tests.** `tests/unit/test_drawdown.py` — value `−1/3` on a hand series, correct
peak/trough dates, monotone-up series → 0.

---

## 5. Correlation

**Definition.** Pearson correlation of asset return series; plus the set of highly
correlated pairs (a concentration/diversification signal).

**Implementation.** `app/risk/correlation.py` — full matrix + upper-triangle pairs with
`|ρ| ≥ 0.8`.

**Tests.** `tests/unit/test_correlation.py` — perfect (+1) / anti (−1) correlation,
threshold filtering, single-asset 1×1.

---

## 6. Exposure

**Definitions.** `gross = Σ|mv_i|`, `net = Σ mv_i`, weights `w_i = mv_i / gross`
(signed; long-only sums to 1). Net weights `mv_i / net` are used for risk decomposition.

**Implementation.** `app/risk/exposure.py` (pure) and `PortfolioService.value_portfolio`.

**Tests.** `tests/unit/test_exposure.py` — long-only vs long/short, zero-net guard.

---

## 7. Risk Contribution (asset-level)

**Definition.** How much each asset contributes to *portfolio* volatility, accounting
for correlations — not just its standalone risk.

**Method — Euler decomposition of volatility.** With net weights `w` (Σ=1) and return
covariance `Σ`:
```
σ_p   = √(wᵀ Σ w)
MCR_i = (Σ w)_i / σ_p          (marginal)
CCR_i = w_i · MCR_i            (component)      with   Σ_i CCR_i = σ_p
PCR_i = CCR_i / σ_p            (percent)        with   Σ_i PCR_i = 1
```
Because volatility is homogeneous of degree 1 in the weights, Euler's theorem makes the
component contributions **sum exactly to σ_p** — an additive, honest attribution.

**Why this matters.** A high-volatility asset that diversifies can contribute *less*
risk than its weight; a correlated one contributes *more*. In the demo, **NVDA is 39.3%
of the book but 55.5% of the risk** — exactly the kind of insight the GenAI assistant
surfaces.

**Implementation.** `app/analytics/risk_contribution.py`.

**Tests.** `tests/unit/test_risk_contribution.py` — components sum to σ_p (Euler
identity), percents sum to 1, single-asset = 100%, constant-returns = 0.

---

## Edge cases (all raise typed, tested errors)

| Situation | Behavior |
|---|---|
| Empty portfolio | `InsufficientHistoricalData` |
| < 20 overlapping observations | `InsufficientHistoricalData` |
| Held ticker has no price history | `MarketDataNotFound` |
| Constant prices | volatility 0, VaR 0, risk contribution 0 (no error) |
| Single asset | correlation 1×1, weight 1.0, 100% risk contribution |
| `confidence_level ∉ (0,1)` | `ValueError` |

## Persistence & auditability

Every `RiskService.compute_metrics` run appends a `risk_metrics` snapshot (VaR,
parametric VaR, annualized vol, max drawdown, gross/net exposure, confidence level,
date). The append-only history lets the assistant answer *"why did risk increase?"* by
comparing successive snapshots.
```
