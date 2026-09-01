"""Domain exception hierarchy.

Every expected error condition in RiskLens is modelled as a subclass of
``RiskLensError``. This lets the API layer install a *single* exception handler
that maps domain errors to structured HTTP responses (see ``app/main.py``) without
leaking stack traces to clients.

Each exception carries:
    * ``code``        – a stable, machine-readable string (used in the API envelope)
    * ``message``     – a human-readable description
    * ``http_status`` – the HTTP status the API layer should return
    * ``details``     – optional structured context (never contains secrets)
"""

from __future__ import annotations

from typing import Any


class RiskLensError(Exception):
    """Base class for all domain errors raised inside RiskLens."""

    code: str = "RISKLENS_ERROR"
    http_status: int = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the API error envelope shape ``{code, message, details}``."""
        return {"code": self.code, "message": self.message, "details": self.details}


# ── Not-found (404) ─────────────────────────────────────────
class NotFoundError(RiskLensError):
    code = "NOT_FOUND"
    http_status = 404


class MarketDataNotFound(NotFoundError):
    code = "MARKET_DATA_NOT_FOUND"


class PortfolioNotFound(NotFoundError):
    code = "PORTFOLIO_NOT_FOUND"


class PositionNotFound(NotFoundError):
    code = "POSITION_NOT_FOUND"


class AlertNotFound(NotFoundError):
    code = "ALERT_NOT_FOUND"


# ── Validation / bad input (422) ────────────────────────────
class ValidationError(RiskLensError):
    code = "VALIDATION_ERROR"
    http_status = 422


class DataValidationError(ValidationError):
    """Raised when ingested market data fails pipeline validation."""

    code = "DATA_VALIDATION_ERROR"


class InvalidPosition(ValidationError):
    """Raised when a portfolio position is malformed (e.g. non-positive quantity)."""

    code = "INVALID_POSITION"


# ── Conflict / insufficient state (409) ─────────────────────
class ConflictError(RiskLensError):
    code = "CONFLICT"
    http_status = 409


class DuplicateResourceError(ConflictError):
    """Raised when creating a resource that violates a uniqueness constraint."""

    code = "DUPLICATE_RESOURCE"


class InsufficientHistoricalData(ConflictError):
    """Raised when there are too few observations to compute a risk metric."""

    code = "INSUFFICIENT_HISTORICAL_DATA"


# ── Computation failures (500) ──────────────────────────────
class RiskCalculationError(RiskLensError):
    """Raised when a risk computation fails for an unexpected reason."""

    code = "RISK_CALCULATION_ERROR"
    http_status = 500


# ── GenAI / assistant (varies) ──────────────────────────────
class AssistantError(RiskLensError):
    """Raised when the GenAI assistant pipeline fails."""

    code = "ASSISTANT_ERROR"
    http_status = 502
