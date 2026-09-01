"""Repository for risk alerts, with database-enforced de-duplication."""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy import select

from app.db.models import Alert
from app.db.repositories.base import BaseRepository


class AlertRepository(BaseRepository[Alert]):
    model = Alert

    def get_by_dedup_key(self, dedup_key: str) -> Alert | None:
        """Return an existing alert with the given de-duplication key, or ``None``."""
        stmt = select(Alert).where(Alert.dedup_key == dedup_key)
        return self.session.execute(stmt).scalar_one_or_none()

    def create_if_new(
        self,
        *,
        portfolio_id: int,
        alert_type: str,
        severity: str,
        message: str,
        dedup_key: str,
        metric_value: Decimal | None = None,
        threshold: Decimal | None = None,
    ) -> tuple[Alert, bool]:
        """Create an alert unless one with the same ``dedup_key`` already exists.

        Returns ``(alert, created)`` where ``created`` is False when a matching alert
        was already present. This is how the engine avoids duplicate alerts for the
        same logical event.
        """
        existing = self.get_by_dedup_key(dedup_key)
        if existing is not None:
            return existing, False
        alert = self.add(
            Alert(
                portfolio_id=portfolio_id,
                alert_type=alert_type,
                severity=severity,
                message=message,
                dedup_key=dedup_key,
                metric_value=metric_value,
                threshold=threshold,
                acknowledged=False,
            )
        )
        return alert, True

    def list_by_portfolio(
        self, portfolio_id: int, *, acknowledged: bool | None = None
    ) -> list[Alert]:
        """Return alerts for a portfolio, optionally filtered by ack status."""
        stmt = select(Alert).where(Alert.portfolio_id == portfolio_id)
        if acknowledged is not None:
            stmt = stmt.where(Alert.acknowledged == acknowledged)
        stmt = stmt.order_by(Alert.created_at.desc(), Alert.id.desc())
        return list(self.session.execute(stmt).scalars().all())

    def acknowledge(self, alert: Alert) -> Alert:
        """Mark an alert acknowledged."""
        alert.acknowledged = True
        self.session.flush()
        return alert
