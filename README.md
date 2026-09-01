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
* **Phase 10:** FastAPI REST API — 23 endpoints across market data, portfolios, risk, correlation, stress, anomalies, and alerts, with Pydantic validation, a uniform error envelope, and OpenAPI docs at `/docs`.
* **Phase 11:** Streamlit dashboard — Overview, Portfolio, Risk Analytics, Stress Testing, Anomalies, Alerts, and an AI Assistant chat shell, with Plotly visualizations, talking to the API over HTTP.
* **Phase 12:** LangChain tools — nine grounded tools (`get_portfolio_summary`, `get_risk_metrics`, `get_risk_contributions`, `run_stress_test`, `get_correlation_matrix`, `get_drawdown_analysis`, `get_asset_exposure`, `get_anomalies`, `get_alerts`) wrapping the services, plus an LLM provider abstraction that runs a deterministic mock with no API key. The LLM can only report numbers a tool returns.
* **Phase 13:** LangGraph assistant — a typed-state graph (classify → plan → retrieve → explain → validate) that answers grounded questions via `POST /assistant/query` and the dashboard chat. Works offline (deterministic renderer) or with a real LLM; a validation node flags any ungrounded figure.
* **Phase 14:** testing & hardening — **197 tests at 97% coverage**, including a consolidated end-to-end demo test and cross-layer consistency checks.
* **Phase 15:** Dockerization + docs — `docker compose up --build` starts postgres + API + dashboard (wait-for-DB, auto-migrate, optional demo seed); design docs in [`docs/`](docs/).
* **Phase 16:** end-to-end demo runner (`make demo`) + [interview guide](docs/interview-guide.md). **Project complete.**

## Documentation

- [Architecture](docs/architecture.md) — layers, decisions, scalability, failure modes
- [Database design](docs/database-design.md) — schema, constraints, rationale
- [Risk methodology](docs/risk-methodology.md) — every metric: definition, formula, test
- [GenAI design](docs/genai-design.md) — LangGraph, tools, grounding, safety

### Testing

```bash
make test                                  # 197 tests
.venv/bin/pytest --cov=app --cov-report=term-missing   # coverage
make test-unit / make test-integration     # by marker
```

Unit tests pin the risk math against hand-computed values (VaR, volatility, drawdown,
the Euler risk-contribution identity, stress P&L) and the assistant's grounding
validator; integration tests cover the database, every API endpoint, the pipelines, and
a full **end-to-end demo** that asserts the assistant's VaR equals the risk engine's.

### Dashboard

```bash
make run          # terminal 1: API on :8000
make dashboard    # terminal 2: dashboard on :8501
```

Then in the sidebar: connect, click **Create demo 'Tech Growth' portfolio** (after
`make load-data`), and explore. Set `RISKLENS_API_URL` to point at a non-local API.

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

### One-command demo

See the entire pipeline run and print every step (self-contained — uses a local SQLite
file, no Postgres needed):

```bash
make demo
```

It loads ~11k synthetic bars, builds the Tech Growth portfolio, computes risk, detects
anomalies, raises alerts, runs a stress scenario, and prints the grounded AI assistant
answering five questions.

Then check the health endpoint:

```bash
curl http://localhost:8000/api/v1/health
```

Expected:

```json
{"status": "ok", "app": "RiskLens", "environment": "development", "version": "0.1.0"}
```

## Docker

One command brings up **PostgreSQL + API + dashboard**. The API entrypoint waits for
the database, applies migrations, and (with `SEED_ON_START=true`) loads synthetic data
and a demo **Tech Growth** portfolio on first boot:

```bash
docker compose up --build
```

- API + docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501

Runs with **no API key** (mock assistant). To use a real model, set `LLM_PROVIDER=openai`
and `OPENAI_API_KEY` in the `api` service environment. Migrations can also be run
manually inside the container: `docker compose exec api alembic upgrade head`.

## Configuration

All configuration is environment-driven. Copy `.env.example` to `.env` and edit.
The application runs **without an LLM API key** by keeping `LLM_PROVIDER=mock`.

## Running tests

```bash
make test          # everything (197 tests)
make test-unit     # fast unit tests only
```

---

## Sample portfolio (the demo)

**Tech Growth** — AAPL 100 · MSFT 80 · NVDA 50 · AMZN 60 · GOOGL 40. On the seeded
dataset this values at **$184,657** with headline risk:

| Metric | Value |
|---|---|
| Annualized volatility | 21.2% |
| 95% / 99% 1-day VaR | $3,731 / $5,715 |
| Max drawdown | −37.3% (2022-10-05 → 2024-01-09) |
| Top risk contributor | **NVDA — 39.3% weight → 55.5% of risk** |
| 25% tech-selloff loss | −$46,164 (−25.0%) |

## Example assistant queries (grounded, real output)

> **Q:** Explain my 99% VaR in simple terms.
> **A:** The 99% 1-day historical VaR is 3.09% (~$5,715 of a $184,657 portfolio)…
> a one-day loss greater than about $5,715 occurred in roughly 1% of days. It is not a
> maximum… parametric estimate is $5,585; it can understate risk when returns have fat
> tails. *Not financial advice.*

> **Q:** Which asset contributes most to portfolio risk?
> **A:** The largest driver is NVDA, contributing **55.5%** of total portfolio volatility
> (weight 39.3%)… risk contribution accounts for correlations, so a volatile, highly
> weighted name can contribute more risk than its weight alone suggests.

Every number in every answer originates from the analytical engine — the assistant
cannot fabricate a metric (see [GenAI design](docs/genai-design.md)).

## Limitations

- **Synthetic data** only (clearly labeled) — realistic characteristics, not real prices.
- `adjusted_close == close` (no dividends/splits modeled).
- VaR/volatility assume i.i.d.-ish returns (√time scaling); no VaR backtest yet.
- Fixed-contamination anomaly detection flags ~that fraction of observations.
- "Why did risk *change*?" is answered as current drivers; a true snapshot diff over the
  append-only `risk_metrics` history is a designed-in future feature.

## Future improvements

- VaR backtesting (Kupiec POF + Christoffersen independence) surfaced in the UI.
- Real dividend/split-adjusted data ingestion.
- Background workers for large ingestion + heavy risk compute; Redis cache.
- Real-time risk via streaming prices + EWMA covariance + WebSockets.
- API authentication/authorization + rate limiting.
- Snapshot-diff explanations ("risk rose because …").

## Disclaimer

RiskLens is a portfolio/research/demo project on **synthetic data**. It is **not financial
advice** and must not be used for real trading or investment decisions. VaR is a
statistical estimate, not a guarantee of maximum loss.

## Interview guide

See [`docs/interview-guide.md`](docs/interview-guide.md) for defensible answers to the
questions this project invites (Why FastAPI/PostgreSQL/historical VaR/Isolation Forest/
LangGraph, how VaR works, backtesting, preventing hallucinations, scaling, AWS deployment,
and what I'd change for production).
