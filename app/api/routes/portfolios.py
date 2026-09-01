"""Portfolio and position routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from app.api.dependencies import get_portfolio_service
from app.schemas.portfolio import (
    PortfolioCreate,
    PortfolioRead,
    PortfolioValuation,
    PositionCreate,
    PositionRead,
    PositionUpdate,
)
from app.services.portfolio_service import PortfolioService

router = APIRouter(prefix="/portfolios", tags=["portfolios"])


@router.post("", response_model=PortfolioRead, status_code=status.HTTP_201_CREATED)
def create_portfolio(
    payload: PortfolioCreate,
    service: PortfolioService = Depends(get_portfolio_service),
) -> PortfolioRead:
    """Create a new portfolio."""
    return PortfolioRead.model_validate(service.create_portfolio(payload))


@router.get("", response_model=list[PortfolioRead])
def list_portfolios(
    service: PortfolioService = Depends(get_portfolio_service),
) -> list[PortfolioRead]:
    """List all portfolios."""
    return [PortfolioRead.model_validate(p) for p in service.list_portfolios()]


@router.get("/{portfolio_id}", response_model=PortfolioRead)
def get_portfolio(
    portfolio_id: int,
    service: PortfolioService = Depends(get_portfolio_service),
) -> PortfolioRead:
    """Get a portfolio with its positions."""
    return PortfolioRead.model_validate(service.get_portfolio(portfolio_id))


@router.delete(
    "/{portfolio_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response
)
def delete_portfolio(
    portfolio_id: int,
    service: PortfolioService = Depends(get_portfolio_service),
) -> Response:
    """Delete a portfolio and its positions."""
    service.delete_portfolio(portfolio_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{portfolio_id}/valuation", response_model=PortfolioValuation)
def get_valuation(
    portfolio_id: int,
    service: PortfolioService = Depends(get_portfolio_service),
) -> PortfolioValuation:
    """Mark the portfolio to market: holdings, weights, exposure, and P&L."""
    return service.value_portfolio(portfolio_id)


@router.post(
    "/{portfolio_id}/positions",
    response_model=PositionRead,
    status_code=status.HTTP_201_CREATED,
)
def add_position(
    portfolio_id: int,
    payload: PositionCreate,
    service: PortfolioService = Depends(get_portfolio_service),
) -> PositionRead:
    """Add or replace a position in a portfolio."""
    return PositionRead.model_validate(service.add_position(portfolio_id, payload))


@router.patch("/{portfolio_id}/positions/{position_id}", response_model=PositionRead)
def update_position(
    portfolio_id: int,
    position_id: int,
    payload: PositionUpdate,
    service: PortfolioService = Depends(get_portfolio_service),
) -> PositionRead:
    """Update a position's quantity and/or average price."""
    return PositionRead.model_validate(
        service.update_position(portfolio_id, position_id, payload)
    )


@router.delete(
    "/{portfolio_id}/positions/{position_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def remove_position(
    portfolio_id: int,
    position_id: int,
    service: PortfolioService = Depends(get_portfolio_service),
) -> Response:
    """Remove a position from a portfolio."""
    service.remove_position(portfolio_id, position_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
