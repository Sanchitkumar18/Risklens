"""Pure alert-rule evaluation.

Given already-computed analytics (risk report, anomaly scan, stress results) and a set
of thresholds, produce a list of :class:`AlertCandidate` — with no I/O. The service
layer persists candidates and de-duplicates them via each candidate's ``dedup_key``.

Severity is graded by the **breach ratio** ``value / threshold``:

    ratio ≥ 1.50 → CRITICAL
    ratio ≥ 1.25 → HIGH
    ratio ≥ 1.00 → MEDIUM
    ratio ≥ 0.90 → LOW      (approaching the limit, not yet breached)
    else         → no alert
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.enums import AlertSeverity, AlertType

_WARN_RATIO = 0.90


@dataclass
class AlertCandidate:
    alert_type: str
    severity: str
    message: str
    metric_value: float
    threshold: float
    dedup_key: str
    ticker: str | None = None


def grade_severity(value: float, threshold: float, warn_ratio: float = _WARN_RATIO) -> str | None:
    """Return a severity for a breach ratio, or ``None`` if well within limits."""
    if threshold <= 0:
        return None
    ratio = value / threshold
    if ratio >= 1.50:
        return AlertSeverity.CRITICAL.value
    if ratio >= 1.25:
        return AlertSeverity.HIGH.value
    if ratio >= 1.00:
        return AlertSeverity.MEDIUM.value
    if ratio >= warn_ratio:
        return AlertSeverity.LOW.value
    return None


def _key(portfolio_id: int, kind: str, as_of, *extra: str) -> str:
    parts = [str(portfolio_id), kind, *extra, str(as_of)]
    return ":".join(parts)


def build_alert_candidates(
    *,
    portfolio_id: int,
    as_of,
    report,             # app.schemas.risk.RiskReport
    anomalies=None,     # app.schemas.anomaly.AnomalyScanResult | None
    stress_results=None,  # list[app.schemas.stress.StressResult] | None
    thresholds,         # app.schemas.alert.AlertThresholds
) -> list[AlertCandidate]:
    """Evaluate every configured threshold and return the resulting candidates."""
    candidates: list[AlertCandidate] = []

    # ── VaR (at the report's confidence level) ──────────────
    headline = next(
        (v for v in report.var_historical if v.confidence_level == report.confidence_level),
        None,
    )
    if headline is not None:
        sev = grade_severity(headline.var_fraction, thresholds.var_limit)
        if sev:
            candidates.append(
                AlertCandidate(
                    alert_type=AlertType.VAR_BREACH.value,
                    severity=sev,
                    message=(
                        f"{report.confidence_level:.0%} 1-day VaR is {headline.var_fraction:.2%} "
                        f"(${headline.var_value:,.0f}), limit {thresholds.var_limit:.2%}."
                    ),
                    metric_value=headline.var_fraction,
                    threshold=thresholds.var_limit,
                    dedup_key=_key(portfolio_id, "VAR", as_of),
                )
            )

    # ── Volatility ──────────────────────────────────────────
    sev = grade_severity(report.volatility_annualized, thresholds.volatility_limit)
    if sev:
        candidates.append(
            AlertCandidate(
                alert_type=AlertType.VOLATILITY_BREACH.value,
                severity=sev,
                message=(
                    f"Annualized volatility is {report.volatility_annualized:.1%}, "
                    f"limit {thresholds.volatility_limit:.1%}."
                ),
                metric_value=report.volatility_annualized,
                threshold=thresholds.volatility_limit,
                dedup_key=_key(portfolio_id, "VOL", as_of),
            )
        )

    # ── Max drawdown (magnitude) ────────────────────────────
    dd = abs(report.drawdown.max_drawdown)
    sev = grade_severity(dd, thresholds.drawdown_limit)
    if sev:
        candidates.append(
            AlertCandidate(
                alert_type=AlertType.DRAWDOWN_BREACH.value,
                severity=sev,
                message=(
                    f"Max drawdown is {report.drawdown.max_drawdown:.1%}, "
                    f"limit {thresholds.drawdown_limit:.1%}."
                ),
                metric_value=dd,
                threshold=thresholds.drawdown_limit,
                dedup_key=_key(portfolio_id, "DD", as_of),
            )
        )

    # ── Single-name concentration ───────────────────────────
    for ticker, weight in report.weights.items():
        sev = grade_severity(abs(weight), thresholds.max_single_weight)
        if sev:
            candidates.append(
                AlertCandidate(
                    alert_type=AlertType.CONCENTRATION.value,
                    severity=sev,
                    message=(
                        f"{ticker} weight is {weight:.1%}, "
                        f"single-name limit {thresholds.max_single_weight:.1%}."
                    ),
                    metric_value=abs(weight),
                    threshold=thresholds.max_single_weight,
                    dedup_key=_key(portfolio_id, "CONC", as_of, ticker),
                    ticker=ticker,
                )
            )

    # ── Anomalies (per ticker, worst score) ─────────────────
    if anomalies is not None and anomalies.anomalies:
        worst_by_ticker: dict[str, float] = {}
        for a in anomalies.anomalies:
            worst_by_ticker[a.ticker] = max(worst_by_ticker.get(a.ticker, 0.0), a.anomaly_score)
        for ticker, score in worst_by_ticker.items():
            sev = grade_severity(score, thresholds.anomaly_score_limit)
            if sev:
                candidates.append(
                    AlertCandidate(
                        alert_type=AlertType.ANOMALY.value,
                        severity=sev,
                        message=(
                            f"{ticker} shows anomalous behavior "
                            f"(score {score:.2f}, limit {thresholds.anomaly_score_limit:.2f})."
                        ),
                        metric_value=score,
                        threshold=thresholds.anomaly_score_limit,
                        dedup_key=_key(portfolio_id, "ANOMALY", as_of, ticker),
                        ticker=ticker,
                    )
                )

    # ── Stress loss (worst scenario) ────────────────────────
    if stress_results:
        worst = min(stress_results, key=lambda s: s.pct_loss)  # most negative
        loss = abs(worst.pct_loss)
        sev = grade_severity(loss, thresholds.stress_loss_limit)
        if sev:
            candidates.append(
                AlertCandidate(
                    alert_type=AlertType.STRESS_LOSS.value,
                    severity=sev,
                    message=(
                        f"Scenario '{worst.scenario_name}' loses {worst.pct_loss:.1%} "
                        f"(${worst.total_loss:,.0f}), limit {thresholds.stress_loss_limit:.1%}."
                    ),
                    metric_value=loss,
                    threshold=thresholds.stress_loss_limit,
                    dedup_key=_key(portfolio_id, "STRESS", as_of, worst.scenario_name),
                )
            )

    return candidates
