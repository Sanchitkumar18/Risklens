"""Coverage for infrastructure edges: DB session lifecycle, provider, repo empties."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.db import database
from app.llm.providers import get_chat_model


@pytest.mark.unit
def test_get_db_rolls_back_and_closes(db_engine, monkeypatch):
    factory = sessionmaker(bind=db_engine)
    monkeypatch.setattr(database, "get_session_factory", lambda: factory)

    gen = database.get_db()
    session = next(gen)
    assert session.is_active
    # Exception inside the request must trigger rollback + close, then re-raise.
    with pytest.raises(ValueError):
        gen.throw(ValueError("boom"))


@pytest.mark.unit
def test_build_engine_sqlite():
    engine = database._build_engine("sqlite://")
    assert engine.dialect.name == "sqlite"
    engine.dispose()


@pytest.mark.unit
def test_real_model_constructed_when_enabled():
    settings = Settings(llm_provider="openai", openai_api_key="sk-test", model_name="gpt-4o-mini")
    model = get_chat_model(settings)
    from langchain_openai import ChatOpenAI

    assert isinstance(model, ChatOpenAI)


@pytest.mark.integration
def test_repo_empty_input_edges(db_session):
    from app.db.repositories.anomaly_repo import AnomalyRepository
    from app.db.repositories.market_data_repo import MarketDataRepository

    md = MarketDataRepository(db_session)
    assert md.latest_bars([]) == {}
    assert md.get_for_tickers([]) == []
    assert md.bulk_upsert([]) == 0
    assert md.distinct_tickers() == []

    anon = AnomalyRepository(db_session)
    assert anon.bulk_add([]) == 0
    assert anon.list_by_ticker("AAPL") == []
