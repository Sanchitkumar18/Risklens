"""Assistant service — the entrypoint that runs the LangGraph assistant.

Verifies the portfolio exists, builds a portfolio-scoped grounded toolkit and the
configured model, runs the graph, and returns a grounded :class:`AssistantResponse`.
Works fully offline (deterministic mock) when no LLM API key is set.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm.graph import build_assistant_graph
from app.llm.providers import get_chat_model, llm_enabled
from app.llm.state import AssistantState
from app.llm.tools import RiskLensToolkit
from app.schemas.assistant import AssistantQuery, AssistantResponse
from app.services.portfolio_service import PortfolioService

logger = get_logger("risklens.assistant")


class AssistantService:
    """Answer grounded natural-language questions about a portfolio's risk."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def query(self, request: AssistantQuery) -> AssistantResponse:
        """Run the assistant graph for a question and return a grounded answer."""
        PortfolioService(self.session).get_portfolio(request.portfolio_id)  # 404 if missing

        settings = get_settings()
        toolkit = RiskLensToolkit(self.session, request.portfolio_id)
        model = get_chat_model(settings) if llm_enabled(settings) else None

        graph = build_assistant_graph(toolkit, settings, model)
        initial: AssistantState = {
            "question": request.question,
            "portfolio_id": request.portfolio_id,
            "confidence": request.confidence,
            "warnings": [],
        }
        result = graph.invoke(initial)

        logger.info(
            "assistant answered",
            extra={
                "portfolio_id": request.portfolio_id,
                "intent": result.get("intent"),
                "tools_used": result.get("tools_used", []),
                "llm": bool(model),
            },
        )
        return AssistantResponse(
            portfolio_id=request.portfolio_id,
            question=request.question,
            intent=result.get("intent", "risk_summary"),
            answer=result.get("final_answer", ""),
            tools_used=result.get("tools_used", []),
            grounded_data=result.get("tool_results", {}),
            warnings=result.get("warnings", []),
        )
