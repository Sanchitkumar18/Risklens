"""Integration tests for the persistence layer (models + repositories).

Run against an in-memory SQLite database created from the ORM metadata.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.exc import IntegrityError

from app.db.repositories.alert_repo import AlertRepository
from app.db.repositories.anomaly_repo import AnomalyRepository
from app.db.repositories.market_data_repo import MarketDataRepository
from app.db.repositories.portfolio_repo import PortfolioRepository
from app.db.repositories.position_repo import PositionRepository
from app.db.repositories.risk_metric_repo import RiskMetricRepository


def _bar(ticker: str, d: date, close: float) -> dict:
    return {
        "ticker": ticker,
        "date": d,
        "open": Decimal(str(close)),
        "high": Decimal(str(close + 1)),
        "low": Decimal(str(close - 1)),
        "close": Decimal(str(close)),
        "adjusted_close": Decimal(str(close)),
        "volume": 1_000_000,
    }


# ── market_data ─────────────────────────────────────────────
@pytest.mark.integration
def test_market_data_upsert_is_idempotent(db_session) -> None:
    repo = MarketDataRepository(db_session)
    rows = [_bar("AAPL", date(2024, 1, 2), 100.0), _bar("AAPL", date(2024, 1, 3), 101.0)]

    repo.bulk_upsert(rows)
    db_session.commit()
    assert repo.count_for_ticker("AAPL") == 2

    # Re-ingest the same dates with a changed close → update, not duplicate.
    rows[0]["close"] = Decimal("123.0")
    repo.bulk_upsert(rows)
    db_session.commit()

    assert repo.count_for_ticker("AAPL") == 2
    bars = repo.get_by_ticker("AAPL", date(2024, 1, 2), date(2024, 1, 2))
    assert bars[0].close == Decimal("123.0")


@pytest.mark.integration
def test_market_data_range_and_multi_ticker_query(db_session) -> None:
    repo = MarketDataRepository(db_session)
    repo.bulk_upsert(
        [
            _bar("AAPL", date(2024, 1, 2), 100.0),
            _bar("AAPL", date(2024, 1, 3), 101.0),
            _bar("MSFT", date(2024, 1, 2), 200.0),
        ]
    )
    db_session.commit()

    assert repo.distinct_tickers() == ["AAPL", "MSFT"]
    assert len(repo.get_for_tickers(["AAPL", "MSFT"])) == 3
    assert len(repo.get_by_ticker("AAPL", date(2024, 1, 3), None)) == 1


@pytest.mark.integration
def test_market_data_high_low_check_constraint(db_session) -> None:
    repo = MarketDataRepository(db_session)
    bad = _bar("AAPL", date(2024, 1, 2), 100.0)
    bad["high"] = Decimal("50")  # high < low violates the check constraint
    bad["low"] = Decimal("99")
    with pytest.raises(IntegrityError):
        repo.bulk_upsert([bad])
        db_session.flush()


# ── portfolios & positions ──────────────────────────────────
@pytest.mark.integration
def test_portfolio_unique_name(db_session) -> None:
    repo = PortfolioRepository(db_session)
    repo.create("Tech Growth")
    db_session.commit()
    with pytest.raises(IntegrityError):
        repo.create("Tech Growth")
        db_session.flush()


@pytest.mark.integration
def test_position_upsert_and_unique(db_session) -> None:
    pf = PortfolioRepository(db_session).create("Tech Growth")
    db_session.commit()

    pos_repo = PositionRepository(db_session)
    pos_repo.upsert(pf.id, "AAPL", Decimal("100"), Decimal("150"))
    db_session.commit()

    # Upsert same ticker updates in place (no second row).
    pos_repo.upsert(pf.id, "AAPL", Decimal("120"), Decimal("155"))
    db_session.commit()

    positions = pos_repo.list_by_portfolio(pf.id)
    assert len(positions) == 1
    assert positions[0].quantity == Decimal("120")


@pytest.mark.integration
def test_position_cascade_delete(db_session) -> None:
    pf_repo = PortfolioRepository(db_session)
    pos_repo = PositionRepository(db_session)
    pf = pf_repo.create("Doomed")
    db_session.commit()
    pos_repo.upsert(pf.id, "NVDA", Decimal("10"), Decimal("500"))
    db_session.commit()

    pf_repo.delete(pf)
    db_session.commit()

    assert pos_repo.list_by_portfolio(pf.id) == []


# ── risk_metrics ────────────────────────────────────────────
@pytest.mark.integration
def test_risk_metric_latest(db_session) -> None:
    from app.db.models import RiskMetric

    pf = PortfolioRepository(db_session).create("PF")
    db_session.commit()
    repo = RiskMetricRepository(db_session)
    repo.add(
        RiskMetric(
            portfolio_id=pf.id, calculation_date=date(2024, 1, 1),
            confidence_level=Decimal("0.95"), var=Decimal("100"),
        )
    )
    repo.add(
        RiskMetric(
            portfolio_id=pf.id, calculation_date=date(2024, 1, 5),
            confidence_level=Decimal("0.95"), var=Decimal("250"),
        )
    )
    db_session.commit()

    latest = repo.latest_for_portfolio(pf.id)
    assert latest is not None and latest.var == Decimal("250")


# ── anomalies ───────────────────────────────────────────────
@pytest.mark.integration
def test_anomaly_bulk_add_with_json_features(db_session) -> None:
    pf = PortfolioRepository(db_session).create("PF")
    db_session.commit()
    repo = AnomalyRepository(db_session)
    n = repo.bulk_add(
        [
            {
                "portfolio_id": pf.id, "ticker": "TSLA", "date": date(2024, 3, 1),
                "anomaly_score": Decimal("-0.42"), "anomaly_type": "return_spike",
                "features": {"daily_return": -0.18, "rolling_vol": 0.05},
            }
        ]
    )
    db_session.commit()
    assert n == 1
    stored = repo.list_by_portfolio(pf.id)[0]
    assert stored.features["daily_return"] == -0.18


# ── alerts (de-duplication) ─────────────────────────────────
@pytest.mark.integration
def test_alert_dedup(db_session) -> None:
    pf = PortfolioRepository(db_session).create("PF")
    db_session.commit()
    repo = AlertRepository(db_session)

    alert1, created1 = repo.create_if_new(
        portfolio_id=pf.id, alert_type="VAR_BREACH", severity="HIGH",
        message="VaR exceeded limit", dedup_key="PF:VAR_BREACH:2024-01-05",
        metric_value=Decimal("12450"), threshold=Decimal("10000"),
    )
    db_session.commit()
    assert created1 is True

    alert2, created2 = repo.create_if_new(
        portfolio_id=pf.id, alert_type="VAR_BREACH", severity="HIGH",
        message="VaR exceeded limit", dedup_key="PF:VAR_BREACH:2024-01-05",
    )
    assert created2 is False
    assert alert2.id == alert1.id
    assert len(repo.list_by_portfolio(pf.id)) == 1


@pytest.mark.integration
def test_alert_acknowledge(db_session) -> None:
    pf = PortfolioRepository(db_session).create("PF")
    db_session.commit()
    repo = AlertRepository(db_session)
    alert, _ = repo.create_if_new(
        portfolio_id=pf.id, alert_type="VOL", severity="LOW",
        message="vol", dedup_key="PF:VOL:2024-01-05",
    )
    db_session.commit()

    repo.acknowledge(alert)
    db_session.commit()
    assert repo.list_by_portfolio(pf.id, acknowledged=False) == []
    assert len(repo.list_by_portfolio(pf.id, acknowledged=True)) == 1


# ── readiness endpoint (DB-backed) ──────────────────────────
@pytest.mark.integration
def test_readiness_endpoint(client_db) -> None:
    resp = client_db.get("/api/v1/health/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready", "database": "ok"}
