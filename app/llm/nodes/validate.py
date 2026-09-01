"""Validation node: the anti-hallucination backstop.

Checks that every number in the draft answer traces back to the tool results. In the
deterministic (mock) path the renderer only uses tool numbers, so this is a no-op; when
a real LLM writes the prose, any figure it invents is flagged as a warning (the answer
is not silently trusted). Tool-level errors are surfaced as warnings too.
"""

from __future__ import annotations

import re

from app.llm.state import AssistantState

# ISO dates are stripped before number extraction so date digits aren't mistaken for
# fabricated metrics.
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUMBER = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def _collect_grounded(obj: object, acc: set[float]) -> None:
    """Recursively collect numeric values (and common rounded/percent forms)."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        v = float(obj)
        for r in (0, 1, 2, 4):
            acc.add(round(v, r))
            acc.add(round(abs(v), r))
            acc.add(round(v * 100, r))
            acc.add(round(abs(v) * 100, r))
    elif isinstance(obj, dict):
        for value in obj.values():
            _collect_grounded(value, acc)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_grounded(item, acc)


def find_ungrounded_numbers(text: str, tool_results: dict) -> list[float]:
    """Return numbers in ``text`` that don't match any value in ``tool_results``."""
    grounded: set[float] = set()
    _collect_grounded(tool_results, grounded)

    cleaned = _ISO_DATE.sub("", text)
    ungrounded: list[float] = []
    for token in _NUMBER.findall(cleaned):
        raw = token.replace("$", "").replace(",", "")
        try:
            x = float(raw)
        except ValueError:
            continue
        if any(round(x, r) in grounded for r in (0, 1, 2, 4)):
            continue
        ungrounded.append(x)
    return ungrounded


def validate_node(state: AssistantState) -> AssistantState:
    warnings = list(state.get("warnings", []))
    answer = state["draft_answer"]
    tool_results = state.get("tool_results", {})

    if not state.get("grounded", False):
        ungrounded = find_ungrounded_numbers(answer, tool_results)
        if ungrounded:
            warnings.append(
                "Some figures could not be verified against tool outputs: "
                + ", ".join(str(x) for x in ungrounded)
            )

    for name, res in tool_results.items():
        if isinstance(res, dict) and res.get("error"):
            warnings.append(f"{name} unavailable: {res['error']}")

    return {"final_answer": answer, "warnings": warnings}
