"""Retrieval node: execute the planned tools against the grounded toolkit.

Each tool call is guarded so a data problem (e.g. insufficient history) becomes a
structured ``{"error": ...}`` in the results — the assistant then says it lacks the
information rather than crashing or inventing an answer.
"""

from __future__ import annotations

from collections.abc import Callable

from app.core.exceptions import RiskLensError
from app.llm.state import AssistantState
from app.llm.tools import RiskLensToolkit


def make_retrieve(toolkit: RiskLensToolkit) -> Callable[[AssistantState], AssistantState]:
    methods = toolkit.as_dict()

    def retrieve(state: AssistantState) -> AssistantState:
        results: dict = {}
        used: list[str] = []
        for tool in state["tool_plan"]:
            fn = methods.get(tool)
            if fn is None:
                continue
            try:
                if tool == "run_stress_test":
                    res = fn(state.get("scenario", "market_crash"))
                elif tool == "get_risk_metrics" and state.get("confidence"):
                    res = fn(state["confidence"])
                else:
                    res = fn()
                results[tool] = res
                used.append(tool)
            except RiskLensError as exc:
                results[tool] = {"error": exc.message}
            except Exception as exc:  # defensive: never let a tool crash the graph
                results[tool] = {"error": str(exc)}
        return {"tool_results": results, "tools_used": used}

    return retrieve
