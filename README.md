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

Implemented incrementally. **Phase 1 (current): project scaffold, configuration,
logging, exception hierarchy, health API, Docker skeleton, first tests.**

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
