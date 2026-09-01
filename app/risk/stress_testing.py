"""Scenario-based stress testing (pure engine).

Applies deterministic price shocks to a portfolio's current holdings and computes the
resulting profit & loss, per asset and in aggregate. The engine is **extensible**: a
scenario is a small declarative spec, and shocks resolve by precedence
(ticker-specific → asset-class → volatility-based → default), so new scenarios need no
new code paths.

    leg_pnl_i = quantity_i · price_i · shock_i          (shock as a signed fraction)
    total_pnl = Σ_i leg_pnl_i
    pct_loss  = total_pnl / portfolio_value_before

A "volatility shock" is expressed as a `k`-sigma adverse move per asset
(`shock_i = −k · σ_i`), so it slots into the same price-shock machinery given a map of
per-asset volatilities supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal

# ── Asset classification (for class-level shocks) ───────────
_TECH = {"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "TSLA", "META", "AVGO", "AMD"}
_BROAD = {"SPY", "QQQ", "IVV", "VOO", "DIA", "VTI"}


def default_classifier(ticker: str) -> str:
    """Map a ticker to an asset class: 'technology', 'broad', or 'equity'."""
    t = ticker.upper()
    if t in _TECH:
        return "technology"
    if t in _BROAD:
        return "broad"
    return "equity"


# ── Scenario specification ──────────────────────────────────
@dataclass(frozen=True)
class ScenarioSpec:
    """Declarative stress scenario.

    Shock resolution precedence (first match wins):
        1. ``ticker_shocks[ticker]``
        2. ``class_shocks[classify(ticker)]``
        3. ``−vol_multiple · σ_ticker``   (if ``vol_multiple`` set and σ known)
        4. ``default_shock``
        5. 0.0
    """

    name: str
    description: str
    ticker_shocks: Mapping[str, float] = field(default_factory=dict)
    class_shocks: Mapping[str, float] = field(default_factory=dict)
    default_shock: float | None = None
    vol_multiple: float | None = None


def resolve_shocks(
    spec: ScenarioSpec,
    tickers: list[str],
    classifier: Callable[[str], str] = default_classifier,
    sigma_map: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Resolve the signed price shock for each ticker per the scenario's precedence."""
    sigma_map = sigma_map or {}
    shocks: dict[str, float] = {}
    for t in tickers:
        if t in spec.ticker_shocks:
            shocks[t] = float(spec.ticker_shocks[t])
        elif classifier(t) in spec.class_shocks:
            shocks[t] = float(spec.class_shocks[classifier(t)])
        elif spec.vol_multiple is not None and t in sigma_map:
            shocks[t] = -float(spec.vol_multiple) * float(sigma_map[t])
        elif spec.default_shock is not None:
            shocks[t] = float(spec.default_shock)
        else:
            shocks[t] = 0.0
    return shocks


# ── Holdings + results ──────────────────────────────────────
@dataclass
class StressHolding:
    ticker: str
    quantity: Decimal
    price: Decimal


@dataclass
class StressLeg:
    ticker: str
    shock: float
    price_before: Decimal
    price_after: Decimal
    value_before: Decimal
    value_after: Decimal
    pnl: Decimal
    pct_of_portfolio: float


@dataclass
class StressOutcome:
    scenario_name: str
    description: str
    portfolio_value_before: Decimal
    portfolio_value_after: Decimal
    total_pnl: Decimal
    total_loss: Decimal          # positive number when the scenario loses money
    pct_loss: float              # signed; negative = loss
    legs: list[StressLeg]
    worst_assets: list[str]


def apply_stress(
    holdings: list[StressHolding],
    shocks: Mapping[str, float],
    scenario_name: str = "custom",
    description: str = "",
) -> StressOutcome:
    """Apply per-ticker shocks to holdings and compute per-leg and total P&L."""
    value_before = Decimal("0")
    value_after = Decimal("0")
    legs: list[StressLeg] = []

    for h in holdings:
        shock = float(shocks.get(h.ticker, 0.0))
        factor = Decimal("1") + Decimal(str(shock))
        price_after = (h.price * factor).quantize(Decimal("0.000001"))
        vb = (h.quantity * h.price).quantize(Decimal("0.01"))
        va = (h.quantity * price_after).quantize(Decimal("0.01"))
        pnl = (va - vb).quantize(Decimal("0.01"))
        value_before += vb
        value_after += va
        legs.append(
            StressLeg(
                ticker=h.ticker, shock=shock,
                price_before=h.price, price_after=price_after,
                value_before=vb, value_after=va, pnl=pnl,
                pct_of_portfolio=0.0,  # filled below once total known
            )
        )

    total_pnl = (value_after - value_before).quantize(Decimal("0.01"))
    pct_loss = float(total_pnl / value_before) if value_before != 0 else 0.0

    for leg in legs:
        leg.pct_of_portfolio = (
            float(leg.pnl / value_before) if value_before != 0 else 0.0
        )

    legs.sort(key=lambda leg: leg.pnl)  # most negative first
    worst = [leg.ticker for leg in legs if leg.pnl < 0][:3]

    return StressOutcome(
        scenario_name=scenario_name,
        description=description,
        portfolio_value_before=value_before.quantize(Decimal("0.01")),
        portfolio_value_after=value_after.quantize(Decimal("0.01")),
        total_pnl=total_pnl,
        total_loss=(-total_pnl if total_pnl < 0 else Decimal("0.00")),
        pct_loss=pct_loss,
        legs=legs,
        worst_assets=worst,
    )


# ── Built-in scenario registry ──────────────────────────────
MARKET_CRASH = "market_crash"
SEVERE_CRASH = "severe_crash"
TECH_SELLOFF = "tech_selloff"
VOLATILITY_SHOCK = "volatility_shock"


def builtin_scenarios() -> dict[str, ScenarioSpec]:
    """Return the built-in stress scenarios keyed by name."""
    return {
        MARKET_CRASH: ScenarioSpec(
            name=MARKET_CRASH,
            description="Broad market crash: all equities fall 20%.",
            default_shock=-0.20,
        ),
        SEVERE_CRASH: ScenarioSpec(
            name=SEVERE_CRASH,
            description="Severe crash: all equities fall 30%.",
            default_shock=-0.30,
        ),
        TECH_SELLOFF: ScenarioSpec(
            name=TECH_SELLOFF,
            description="Technology selloff: tech −25%, everything else −10%.",
            class_shocks={"technology": -0.25},
            default_shock=-0.10,
        ),
        VOLATILITY_SHOCK: ScenarioSpec(
            name=VOLATILITY_SHOCK,
            description="Volatility shock: a 3-sigma adverse daily move per asset.",
            vol_multiple=3.0,
        ),
    }
