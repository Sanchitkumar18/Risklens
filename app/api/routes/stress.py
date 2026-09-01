"""Stress-testing routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_stress_service
from app.risk.stress_testing import MARKET_CRASH
from app.schemas.stress import CustomScenarioRequest, ScenarioInfo, StressResult
from app.services.stress_service import StressService

router = APIRouter(prefix="/portfolios", tags=["stress"])


@router.get("/{portfolio_id}/stress-test/scenarios", response_model=list[ScenarioInfo])
def list_scenarios(
    portfolio_id: int,
    service: StressService = Depends(get_stress_service),
) -> list[ScenarioInfo]:
    """List the available built-in stress scenarios."""
    return service.list_scenarios()


@router.get("/{portfolio_id}/stress-test", response_model=StressResult)
def run_scenario(
    portfolio_id: int,
    scenario: str = Query(default=MARKET_CRASH, description="Built-in scenario name."),
    service: StressService = Depends(get_stress_service),
) -> StressResult:
    """Run a built-in stress scenario against the portfolio."""
    return service.run_builtin(portfolio_id, scenario)


@router.post("/{portfolio_id}/stress-test/custom", response_model=StressResult)
def run_custom(
    portfolio_id: int,
    payload: CustomScenarioRequest,
    service: StressService = Depends(get_stress_service),
) -> StressResult:
    """Run a user-defined stress scenario."""
    return service.run_custom(portfolio_id, payload)
