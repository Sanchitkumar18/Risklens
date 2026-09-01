"""Shared enumerations (stored as strings in the database)."""

from __future__ import annotations

from enum import Enum


class AlertSeverity(str, Enum):
    """Severity of a risk alert, graded by how far a metric breaches its limit."""

    LOW = "LOW"            # approaching the limit (≈90–100%)
    MEDIUM = "MEDIUM"      # breached (100–125%)
    HIGH = "HIGH"          # substantially breached (125–150%)
    CRITICAL = "CRITICAL"  # severely breached (≥150%)


class AlertType(str, Enum):
    """Category of a risk alert."""

    VAR_BREACH = "VAR_BREACH"
    VOLATILITY_BREACH = "VOLATILITY_BREACH"
    DRAWDOWN_BREACH = "DRAWDOWN_BREACH"
    CONCENTRATION = "CONCENTRATION"
    ANOMALY = "ANOMALY"
    STRESS_LOSS = "STRESS_LOSS"
