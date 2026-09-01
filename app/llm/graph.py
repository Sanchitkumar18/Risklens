"""LangGraph assembly for the RiskLens assistant.

Wires the typed-state nodes into a linear graph:

    classify → plan → retrieve → explain → validate → END

The graph is built per request (bound to a portfolio-scoped toolkit and the configured
model). Nodes read/write ``AssistantState``.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.core.config import Settings
from app.llm.nodes.classify import classify_node
from app.llm.nodes.explain import make_explain
from app.llm.nodes.plan import plan_node
from app.llm.nodes.retrieve import make_retrieve
from app.llm.nodes.validate import validate_node
from app.llm.state import AssistantState
from app.llm.tools import RiskLensToolkit


def build_assistant_graph(toolkit: RiskLensToolkit, settings: Settings, model=None):
    """Build and compile the assistant graph for a given portfolio toolkit."""
    graph = StateGraph(AssistantState)

    graph.add_node("classify", classify_node)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", make_retrieve(toolkit))
    graph.add_node("explain", make_explain(settings, model))
    graph.add_node("validate", validate_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "explain")
    graph.add_edge("explain", "validate")
    graph.add_edge("validate", END)

    return graph.compile()
