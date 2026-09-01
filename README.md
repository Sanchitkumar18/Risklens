# RiskLens — Market Risk Analytics & GenAI Assistant

> **Disclaimer:** RiskLens is a portfolio/research/demo project. It uses **synthetic
> market data** and is **not financial advice**. Do not use it for real trading or
> investment decisions.

RiskLens is a market-risk analytics platform that ingests historical market data,
constructs portfolios, computes industry-standard risk metrics (Historical & Parametric
VaR, volatility, max drawdown, correlation, exposure, risk contribution), runs scenario
stress tests, detects anomalies with scikit-learn, raises configurable alerts, and
exposes a **grounded GenAI assistant** (LangChain + LangGraph) that explains risk using
*only* numbers produced by the analytical engine.

This README is expanded across build phases. See the design in
[`docs/`](docs/) (added in later phases) for full methodology and interview notes.

---

## Tech stack

| Layer      | Technology |
|------------|------------|
| API        | FastAPI · Pydantic v2 · Uvicorn |
| Data store | PostgreSQL · SQLAlchemy 2.0 · Alembic |
| Analytics  | Pandas · NumPy · SciPy · scikit-learn · Plotly |
| GenAI      | LangChain · LangGraph (works **offline** with a deterministic mock LLM) |
| Dashboard  | Streamlit |
| Infra      | Docker · Docker Compose · pytest |

## Build phases

Implemented incrementally.

* **Phase 1:** project scaffold, configuration, logging, exception hierarchy, health API, Docker skeleton, first tests.
* **Phase 2:** PostgreSQL schema — SQLAlchemy 2.0 models (7 tables), engine/session management, repository layer, Alembic migrations, DB-backed readiness probe, integration tests.
* **Phase 3:** synthetic market-data generator (multi-factor + GARCH model) and the ingestion pipeline (CSV/DataFrame → normalized, typed rows → idempotent DB upsert).
* **Phase 4:** validation → cleaning → transformation pipeline. Validation quarantines and reports bad rows (never silently dropped); cleaning deduplicates/sorts; transformation derives returns and rolling features for analytics.
* **Phase 5:** portfolio management — portfolio/position CRUD and mark-to-market valuation (holdings, weights, gross/net exposure, unrealized P&L) computed from stored positions × stored prices.
* **Phase 6:** risk engine — Historical & Parametric VaR, volatility, max drawdown, correlation, exposure, and Euler risk contribution, as pure tested functions; `RiskService` computes and persists a full `RiskReport`. See [`docs/risk-methodology.md`](docs/risk-methodology.md).
* **Phase 7:** scenario stress testing — market crash, severe crash, tech selloff, volatility shock, and custom per-ticker/per-class shocks, with per-asset P&L attribution and worst-affected assets. Extensible via a declarative `ScenarioSpec`.
* **Phase 8:** anomaly detection — scikit-learn Isolation Forest over engineered features (return, rolling vol, volume change, distance-from-MA), per ticker, with configurable contamination; anomalies are typed (price move / volatility / volume / trend) and persisted.
* **Phase 9:** alert engine — configurable thresholds on VaR, volatility, drawdown, single-name concentration, anomaly score, and stress loss; severity graded by breach ratio (LOW/MEDIUM/HIGH/CRITICAL), de-duplicated per breach per day, persisted, and acknowledgeable.
* **Phase 10 (current):** FastAPI REST API — 22 endpoints across market data, portfolios, risk, correlation, stress, anomalies, and alerts, with Pydantic validation, a uniform error envelope, and OpenAPI docs at `/docs`.

### API

```bash
make run              # http://localhost:8000/docs  (interactive Swagger UI)
curl http://localhost:8000/api/v1/health
```

Endpoints (prefix `/api/v1`): `POST /market-data/upload`, `GET /market-data/{ticker}`,
`POST|GET /portfolios`, `GET|DELETE /portfolios/{id}`, `GET /portfolios/{id}/valuation`,
`POST|PATCH|DELETE /portfolios/{id}/positions[/{pid}]`, `GET /portfolios/{id}/risk`,
`GET /portfolios/{id}/correlation`, `GET /portfolios/{id}/stress-test[/scenarios]`,
`POST /portfolios/{id}/stress-test/custom`, `GET|POST /portfolios/{id}/anomalies[/scan]`,
`GET|POST /portfolios/{id}/alerts[/scan]`, `POST /portfolios/{id}/alerts/{aid}/acknowledge`.
Errors return `{"error": {"code", "message", "details"}}` with the right HTTP status.

### Data pipeline

```
raw → normalize → VALIDATE (accept/reject + report) → CLEAN (dedup/sort) → upsert → PostgreSQL
                                                                                  ↓
                                          TRANSFORM (returns, rolling vol, MAs) → analytics
```

Validation categorizes issues as **ERROR** (row rejected: null/negative price,
`high < low`, OHLC inconsistency, bad volume, invalid ticker, unparseable/future date)
or **WARNING** (row kept, flagged: abnormal price jump, out-of-bounds price, duplicate
row, missing business day). Every ingestion returns a `ValidationReport` with rows
processed/accepted/rejected and counts by category.

### Sample data

```bash
make sample-data      # writes data/sample/market_data_sample.csv (reproducible, seed=42)
make load-data        # loads it into the database (run make migrate first)
```

The generator produces ~11k daily bars across 7 assets (SPY + 6 tech names) with
realistic **trends, volatility clustering, cross-asset correlation, tail-event crashes,
and single-name anomalies**. It is **synthetic/demo data only** — not real market data.
The ingestion layer also accepts real vendor CSVs (common column aliases like
`Adj Close` are normalized automatically).

### Database migrations

```bash
# Point DATABASE_URL at your Postgres (see .env), then:
make migrate          # alembic upgrade head
make migrate-down     # roll back one revision
make revision M="add something"   # autogenerate a new migration
```

The schema also runs on SQLite, which is what the in-memory integration test suite
uses — so `make test` needs no database server.

---

## Local setup (Phase 1)

Requires **Python 3.11+**.

```bash
# 1. Create venv + install deps
make install

# 2. Run the test suite
make test

# 3. Run the API (http://localhost:8000/docs)
make run
```

Then check the health endpoint:

```bash
curl http://localhost:8000/api/v1/health
```

Expected:

```json
{"status": "ok", "app": "RiskLens", "environment": "development", "version": "0.1.0"}
```

## Docker (Phase 1)

Brings up PostgreSQL + the API (the API does not yet use the DB — that lands in the
persistence phase):

```bash
cp .env.example .env
docker compose up --build
```

## Configuration

All configuration is environment-driven. Copy `.env.example` to `.env` and edit.
The application runs **without an LLM API key** by keeping `LLM_PROVIDER=mock`.

## Running tests

```bash
make test          # everything
make test-unit     # fast unit tests only
```
