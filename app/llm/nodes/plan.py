"""Planning node: map an intent to the ordered tools to call."""

from __future__ import annotations

from app.llm.nodes import classify as c
from app.llm.state import AssistantState

# Intent → tool method names (executed by the retrieve node).
_PLANS: dict[str, list[str]] = {
    c.RISK_SUMMARY: ["get_portfolio_summary", "get_risk_metrics", "get_risk_contributions", "get_alerts"],
    c.VAR_EXPLAIN: ["get_risk_metrics"],
    c.CONTRIBUTION: ["get_risk_contributions", "get_portfolio_summary"],
    c.DRAWDOWN: ["get_drawdown_analysis"],
    c.CORRELATION: ["get_correlation_matrix"],
    c.EXPOSURE: ["get_asset_exposure"],
    c.STRESS: ["run_stress_test"],
    c.ANOMALIES: ["get_anomalies"],
    c.ALERTS: ["get_alerts"],
    c.PORTFOLIO_SUMMARY: ["get_portfolio_summary"],
}


def choose_scenario(question: str) -> str:
    """Pick a built-in stress scenario from the question text."""
    q = question.lower()
    if "tech" in q or "technology" in q:
        return "tech_selloff"
    if "severe" in q or "30%" in q or "-30" in q:
        return "severe_crash"
    if "volatil" in q or "vol shock" in q:
        return "volatility_shock"
    return "market_crash"


def plan_node(state: AssistantState) -> AssistantState:
    intent = state["intent"]
    plan = _PLANS.get(intent, _PLANS[c.RISK_SUMMARY])
    update: AssistantState = {"tool_plan": plan}
    if intent == c.STRESS:
        update["scenario"] = choose_scenario(state["question"])
    return update
