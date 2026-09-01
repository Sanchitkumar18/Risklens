"""Explanation node: turn grounded tool results into prose.

Two paths:
* **Deterministic (mock / no API key)** — a template renderer builds the answer using
  ONLY numbers from the tool results, so grounding is guaranteed offline.
* **LLM (when configured)** — the chat model writes the prose under a strict grounding
  system prompt, with the tool results supplied as context; the validate node then
  checks it.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import Settings
from app.llm.nodes import classify as c
from app.llm.prompts import DISCLAIMER, EXPLAIN_INSTRUCTIONS, SYSTEM_PROMPT
from app.llm.state import AssistantState

_NOT_ADVICE = "This is analytics on synthetic/demo data, not financial advice."


# ── formatting helpers ──────────────────────────────────────
def _money(v: float | None) -> str:
    return f"${v:,.0f}" if v is not None else "n/a"


def _pct(v: float | None, dp: int = 1) -> str:
    return f"{v * 100:.{dp}f}%" if v is not None else "n/a"


def _has(results: dict, tool: str) -> bool:
    return tool in results and not (isinstance(results[tool], dict) and results[tool].get("error"))


# ── deterministic renderer ──────────────────────────────────
def render_answer(intent: str, question: str, results: dict, confidence: float | None) -> str:
    parts: list[str] = []
    conf = confidence or 0.95

    if intent == c.VAR_EXPLAIN and _has(results, "get_risk_metrics"):
        m = results["get_risk_metrics"]
        target = m.get("confidence_level", conf)
        vh = next((v for v in m["var_historical"] if v["confidence_level"] == target), m["var_historical"][-1])
        parts.append(
            f"The {_pct(vh['confidence_level'],0)} 1-day historical VaR is {_pct(vh['var_fraction'],2)} "
            f"(~{_money(vh['var_value'])} of a {_money(m['portfolio_value'])} portfolio)."
        )
        parts.append(
            f"Interpretation: based on {m['observations']} historical daily returns, a one-day loss greater "
            f"than about {_money(vh['var_value'])} occurred in roughly {(1-vh['confidence_level'])*100:.0f}% of days. "
            "It is not a maximum — larger losses can and do happen beyond this threshold."
        )
        if m.get("var_parametric"):
            parts.append(
                f"For comparison, the parametric (Normal) estimate is {_money(m['var_parametric']['var_value'])}; "
                "it can understate risk when returns have fat tails."
            )

    elif intent == c.CONTRIBUTION and _has(results, "get_risk_contributions"):
        rc = results["get_risk_contributions"]
        top = rc.get("top_contributor")
        if top:
            parts.append(
                f"The largest driver of portfolio risk is {top['ticker']}, contributing "
                f"{_pct(top['percent'])} of total portfolio volatility (weight {_pct(top['weight'])})."
            )
        ranked = ", ".join(f"{x['ticker']} {_pct(x['percent'])}" for x in rc["contributions"][:5])
        parts.append(f"Full ranking by risk contribution: {ranked}.")
        parts.append(
            "Interpretation: risk contribution accounts for correlations, so a volatile, highly weighted "
            "name can contribute more risk than its capital weight alone suggests."
        )

    elif intent == c.DRAWDOWN and _has(results, "get_drawdown_analysis"):
        d = results["get_drawdown_analysis"]
        parts.append(
            f"The maximum drawdown is {_pct(d['max_drawdown'])}, from a peak on {d['peak_date']} "
            f"to a trough on {d['trough_date']}."
        )
        parts.append("This is the worst peak-to-trough decline in the reconstructed portfolio value over the sample.")

    elif intent == c.CORRELATION and _has(results, "get_correlation_matrix"):
        cm = results["get_correlation_matrix"]
        pairs = cm.get("high_correlation_pairs", [])
        if pairs:
            listed = ", ".join(f"{p['ticker_a']}–{p['ticker_b']} ({p['correlation']:.2f})" for p in pairs)
            parts.append(f"Highly correlated pairs (|ρ| ≥ 0.8): {listed}.")
        else:
            parts.append("No asset pairs exceed a correlation of 0.8; the book is reasonably diversified across names.")
        parts.append("Higher correlations reduce diversification benefit and concentrate risk.")

    elif intent == c.EXPOSURE and _has(results, "get_asset_exposure"):
        e = results["get_asset_exposure"]
        parts.append(
            f"Gross exposure is {_money(e['gross_exposure'])} and net exposure {_money(e['net_exposure'])}."
        )
        if e.get("largest_position"):
            lp = e["largest_position"]
            parts.append(f"The largest single position is {lp['ticker']} at {_pct(lp['weight'])} of the book.")

    elif intent == c.STRESS and _has(results, "run_stress_test"):
        s = results["run_stress_test"]
        parts.append(
            f"Under the '{s['scenario']}' scenario ({s['description']}), the portfolio would lose "
            f"{_money(s['total_loss'])} ({_pct(s['pct_loss'])})."
        )
        if s.get("worst_assets"):
            parts.append(f"The worst-affected assets are {', '.join(s['worst_assets'])}.")
        parts.append("This is a hypothetical, instantaneous shock — actual outcomes depend on how a real event unfolds.")

    elif intent == c.ANOMALIES and _has(results, "get_anomalies"):
        a = results["get_anomalies"]
        if a["count"] == 0:
            parts.append("No unusual market movements were detected for the portfolio's assets.")
        else:
            top = a["anomalies"][:3]
            listed = ", ".join(f"{x['ticker']} on {x['date']} ({x['type']})" for x in top)
            parts.append(f"Detected {a['count']} anomalous observation(s); most notable: {listed}.")
            parts.append("Anomalies are unusual combinations of return, volatility and volume — worth review, not automatically losses.")

    elif intent == c.ALERTS and _has(results, "get_alerts"):
        al = results["get_alerts"]
        if al["count"] == 0:
            parts.append("There are no active risk alerts for this portfolio.")
        else:
            listed = "; ".join(f"[{x['severity']}] {x['message']}" for x in al["alerts"][:5])
            parts.append(f"There are {al['count']} active alert(s): {listed}")

    # risk_summary and portfolio_summary (and fallback)
    if not parts:
        _render_summary(parts, results, conf)

    parts.append(_NOT_ADVICE)
    return " ".join(parts)


def _render_summary(parts: list[str], results: dict, conf: float) -> None:
    if _has(results, "get_portfolio_summary"):
        p = results["get_portfolio_summary"]
        parts.append(
            f"Portfolio '{p['name']}' is worth {_money(p['total_value'])} across {p['num_positions']} positions "
            f"(unrealized P&L {_money(p['unrealized_pnl'])})."
        )
    if _has(results, "get_risk_metrics"):
        m = results["get_risk_metrics"]
        vh = next((v for v in m["var_historical"] if v["confidence_level"] == m.get("confidence_level", conf)), m["var_historical"][-1])
        parts.append(
            f"Annualized volatility is {_pct(m['volatility_annualized'])}, and the "
            f"{_pct(vh['confidence_level'],0)} 1-day VaR is about {_money(vh['var_value'])} "
            f"({_pct(vh['var_fraction'],2)})."
        )
    if _has(results, "get_risk_contributions"):
        top = results["get_risk_contributions"].get("top_contributor")
        if top:
            parts.append(
                f"The main risk driver is {top['ticker']} at {_pct(top['percent'])} of portfolio volatility."
            )
    if _has(results, "get_alerts"):
        al = results["get_alerts"]
        if al["count"]:
            parts.append(f"There are {al['count']} active risk alert(s) to review.")
    if not parts:
        parts.append(
            "I don't have enough data to assess this portfolio's risk yet — "
            "ensure it has positions and sufficient price history."
        )


# ── node factory ────────────────────────────────────────────
def make_explain(settings: Settings, model) -> Callable[[AssistantState], AssistantState]:
    use_llm = model is not None

    def explain(state: AssistantState) -> AssistantState:
        results = state.get("tool_results", {})
        if use_llm:
            try:
                answer = _llm_explain(model, state, results)
                return {"draft_answer": answer, "grounded": False}
            except Exception:
                pass  # fall back to the deterministic renderer
        answer = render_answer(state["intent"], state["question"], results, state.get("confidence"))
        return {"draft_answer": answer, "grounded": True}

    return explain


def _llm_explain(model, state: AssistantState, results: dict) -> str:
    context = json.dumps(results, default=str, indent=2)
    human = (
        f"Question: {state['question']}\n\n"
        f"Tool results (JSON):\n{context}\n\n{EXPLAIN_INSTRUCTIONS}\n\n{DISCLAIMER}"
    )
    resp = model.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human)])
    return resp.content if hasattr(resp, "content") else str(resp)
