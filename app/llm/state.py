"""Typed state for the LangGraph assistant.

Flows through: classify → plan → retrieve → explain → validate. Fields accumulate as
each node runs; ``total=False`` because early nodes populate only part of it.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AssistantState(TypedDict, total=False):
    # inputs
    question: str
    portfolio_id: int
    confidence: float | None
    # classify / plan
    intent: str
    tool_plan: list[str]
    scenario: str
    # retrieve
    tool_results: dict[str, Any]
    tools_used: list[str]
    # explain / validate
    draft_answer: str
    grounded: bool
    final_answer: str
    warnings: list[str]
