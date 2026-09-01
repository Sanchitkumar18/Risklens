"""FastAPI dependency providers.

Centralizes the wiring between the request layer and lower layers so routes stay
thin and services receive their collaborators via dependency injection.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.database import get_db as _get_db
from app.services.alert_service import AlertService
from app.services.anomaly_service import AnomalyService
from app.services.market_data_service import MarketDataService
from app.services.portfolio_service import PortfolioService
from app.services.risk_service import RiskService
from app.services.stress_service import StressService


def get_db_session() -> Iterator[Session]:
    """Yield a database session for the lifetime of a request."""
    yield from _get_db()


def get_market_data_service(db: Session = Depends(get_db_session)) -> MarketDataService:
    return MarketDataService(db)


def get_portfolio_service(db: Session = Depends(get_db_session)) -> PortfolioService:
    return PortfolioService(db)


def get_risk_service(db: Session = Depends(get_db_session)) -> RiskService:
    return RiskService(db)


def get_stress_service(db: Session = Depends(get_db_session)) -> StressService:
    return StressService(db)


def get_anomaly_service(db: Session = Depends(get_db_session)) -> AnomalyService:
    return AnomalyService(db)


def get_alert_service(db: Session = Depends(get_db_session)) -> AlertService:
    return AlertService(db)
