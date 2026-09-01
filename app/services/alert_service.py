"""Alert engine service.

Composes the risk, anomaly, and stress services, evaluates configurable thresholds
(pure rules in ``app.analytics.alert_rules``), and persists severity-graded alerts with
database-enforced de-duplication (one alert per logical breach per day).
"""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy.orm import Session

from app.analytics.alert_rules import build_alert_candidates
from app.core.exceptions import AlertNotFound
from app.core.logging import get_logger
from app.db.repositories.alert_repo import AlertRepository
from app.risk.stress_testing import MARKET_CRASH, SEVERE_CRASH, TECH_SELLOFF
from app.schemas.alert import AlertEvaluationResult, AlertRead, AlertThresholds
from app.services.anomaly_service import AnomalyService
from app.services.risk_service import RiskService
from app.services.stress_service import StressService

logger = get_logger("risklens.alert")

# Built-in scenarios evaluated for the stress-loss limit.
_STRESS_SCENARIOS = (MARKET_CRASH, SEVERE_CRASH, TECH_SELLOFF)


class AlertService:
    """Evaluate thresholds and manage persisted alerts."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.alerts = AlertRepository(session)
        self.risk = RiskService(session)
        self.anomaly = AnomalyService(session)
        self.stress = StressService(session)

    def evaluate(
        self,
        portfolio_id: int,
        thresholds: AlertThresholds | None = None,
        *,
        include_anomalies: bool = True,
        include_stress: bool = True,
    ) -> AlertEvaluationResult:
        """Run the engine: compute analytics, grade breaches, persist new alerts."""
        thresholds = thresholds or AlertThresholds()

        report = self.risk.compute_metrics(portfolio_id, persist=False)
        anomalies = (
            self.anomaly.scan_portfolio(portfolio_id) if include_anomalies else None
        )
        stress_results = (
            [self.stress.run_builtin(portfolio_id, name) for name in _STRESS_SCENARIOS]
            if include_stress
            else None
        )

        candidates = build_alert_candidates(
            portfolio_id=portfolio_id,
            as_of=report.as_of_date,
            report=report,
            anomalies=anomalies,
            stress_results=stress_results,
            thresholds=thresholds,
        )

        created = 0
        persisted: list = []
        by_severity: dict[str, int] = {}
        for c in candidates:
            by_severity[c.severity] = by_severity.get(c.severity, 0) + 1
            alert, is_new = self.alerts.create_if_new(
                portfolio_id=portfolio_id,
                alert_type=c.alert_type,
                severity=c.severity,
                message=c.message,
                dedup_key=c.dedup_key,
                metric_value=Decimal(str(round(c.metric_value, 6))),
                threshold=Decimal(str(round(c.threshold, 6))),
            )
            persisted.append(alert)
            created += int(is_new)
        self.session.commit()

        logger.info(
            "alerts evaluated",
            extra={
                "portfolio_id": portfolio_id,
                "breaches": len(candidates),
                "created": created,
                "by_severity": by_severity,
            },
        )
        return AlertEvaluationResult(
            portfolio_id=portfolio_id,
            as_of_date=report.as_of_date,
            breaches=len(candidates),
            created=created,
            existing=len(candidates) - created,
            by_severity=by_severity,
            alerts=[AlertRead.model_validate(a) for a in persisted],
        )

    def list_alerts(
        self, portfolio_id: int, *, acknowledged: bool | None = None
    ) -> list[AlertRead]:
        """List a portfolio's alerts (optionally filtered by ack status)."""
        rows = self.alerts.list_by_portfolio(portfolio_id, acknowledged=acknowledged)
        return [AlertRead.model_validate(a) for a in rows]

    def acknowledge(self, portfolio_id: int, alert_id: int) -> AlertRead:
        """Acknowledge an alert belonging to a portfolio."""
        alert = self.alerts.get(alert_id)
        if alert is None or alert.portfolio_id != portfolio_id:
            raise AlertNotFound(
                f"Alert {alert_id} not found in portfolio {portfolio_id}.",
                details={"portfolio_id": portfolio_id, "alert_id": alert_id},
            )
        self.alerts.acknowledge(alert)
        self.session.commit()
        return AlertRead.model_validate(alert)
