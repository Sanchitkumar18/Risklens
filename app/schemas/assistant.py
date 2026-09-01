"""Pydantic schemas for the GenAI assistant API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AssistantQuery(BaseModel):
    """A question about a portfolio's risk."""

    portfolio_id: int
    question: str = Field(..., min_length=1, max_length=2000)
    confidence: float | None = Field(default=None, gt=0.0, lt=1.0)


class AssistantResponse(BaseModel):
    """A grounded answer produced from analytical tool results."""

    portfolio_id: int
    question: str
    intent: str = Field(..., description="Classified query intent.")
    answer: str = Field(..., description="Natural-language answer grounded in tool data.")
    tools_used: list[str] = Field(default_factory=list)
    grounded_data: dict[str, Any] = Field(
        default_factory=dict, description="The exact tool outputs the answer is based on."
    )
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = Field(
        default=(
            "Analytics on synthetic/demo data for research purposes only. "
            "VaR is not a maximum-loss guarantee. Not financial advice."
        )
    )
