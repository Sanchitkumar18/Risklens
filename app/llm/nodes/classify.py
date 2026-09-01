"""Intent classification node (deterministic, keyword-based).

A deterministic classifier keeps the assistant reproducible and testable — the same
question always routes to the same tools — and needs no LLM call. (A model-based
classifier could be swapped in behind the same interface.)
"""

from __future__ import annotations

from app.llm.state import AssistantState

# Recognized intents.
RISK_SUMMARY = "risk_summary"
VAR_EXPLAIN = "var_explain"
CONTRIBUTION = "contribution"
DRAWDOWN = "drawdown"
CORRELATION = "correlation"
EXPOSURE = "exposure"
STRESS = "stress"
ANOMALIES = "anomalies"
ALERTS = "alerts"
PORTFOLIO_SUMMARY = "portfolio_summary"

# Checked in order; first match wins.
_RULES: list[tuple[str, tuple[str, ...]]] = [
    (ALERTS, ("alert", "limit breach", "threshold")),
    (ANOMALIES, ("anomal", "unusual", "outlier", "strange move")),
    (STRESS, ("stress", "crash", "scenario", "what if", "happens if", "fall", "drop", "shock", "selloff", "sell-off")),
    (DRAWDOWN, ("drawdown", "largest loss", "worst loss", "peak to trough", "biggest decline")),
    (CORRELATION, ("correlat", "diversif", "move together")),
    (CONTRIBUTION, ("contribut", "which asset", "driving", "driver", "riskiest", "most risk", "biggest risk")),
    (VAR_EXPLAIN, ("value at risk", "var")),
    (EXPOSURE, ("exposure", "weight", "concentrat", "position size", "allocation")),
    (RISK_SUMMARY, ("risky", "risk", "how risk", "summary", "overview", "safe")),
    (PORTFOLIO_SUMMARY, ("portfolio", "holding", "worth", "value")),
]


def classify_intent(question: str) -> str:
    """Return the intent for a question (defaults to a risk summary)."""
    q = question.lower()
    for intent, keywords in _RULES:
        if any(k in q for k in keywords):
            return intent
    return RISK_SUMMARY


def classify_node(state: AssistantState) -> AssistantState:
    return {"intent": classify_intent(state["question"])}
