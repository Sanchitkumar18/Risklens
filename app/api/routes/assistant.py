"""GenAI assistant route."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_db_session
from app.schemas.assistant import AssistantQuery, AssistantResponse
from app.services.assistant_service import AssistantService

router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.post("/query", response_model=AssistantResponse, summary="Ask the risk assistant")
def query(
    payload: AssistantQuery,
    db: Session = Depends(get_db_session),
) -> AssistantResponse:
    """Answer a grounded question about a portfolio's risk.

    The assistant retrieves metrics via analytical tools and explains them; it never
    invents numbers. Runs offline with a deterministic mock when no LLM key is set.
    """
    return AssistantService(db).query(payload)
