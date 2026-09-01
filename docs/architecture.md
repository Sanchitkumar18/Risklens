# RiskLens — Architecture

## Overview

RiskLens is a **layered, service-oriented monolith**. Five layers, each depending only
on the one beneath it:

```
┌──────────────────────────────────────────────────────────────┐
│ PRESENTATION   Streamlit dashboard  ·  FastAPI REST / OpenAPI   │
├──────────────────────────────────────────────────────────────┤
│ APPLICATION    Services (orchestration) + LangGraph assistant  │
│   market_data · portfolio · risk · stress · anomaly · alert    │
│   assistant                                                    │
├──────────────────────────────────────────────────────────────┤
│ DOMAIN (pure)  risk/ (var, vol, drawdown, corr, exposure,      │
│   stress) · analytics/ (isolation forest, risk contribution)   │
│   pipelines/ (validate, clean, transform, synthetic)           │
├──────────────────────────────────────────────────────────────┤
│ PERSISTENCE    Repositories · SQLAlchemy models · Alembic      │
├──────────────────────────────────────────────────────────────┤
│ DATA STORE     PostgreSQL                                      │
└──────────────────────────────────────────────────────────────┘
```

**The core invariant:** `app/risk/`, `app/analytics/`, and `app/pipelines/` are **pure**
— they take Pandas/NumPy in and return numbers/frames out, with no DB, network, config,
or LLM access. Services are the only place DB + domain + config meet. Routes are thin
(parse → call service → serialize). **The LangGraph assistant is just another client of
the services**, via grounded tools — it has no independent path to a number.

## Request lifecycle (API)

```
HTTP → middleware (request id + timing log)
     → route (Pydantic validation)
     → dependency injection (request-scoped DB session + service)
     → service (transaction boundary) → repositories → PostgreSQL
                                      → pure domain functions
     → Pydantic response  |  RiskLensError → single exception handler → {"error": {...}}
```

## Component diagram

```
 Streamlit ──HTTP──► FastAPI ──► services ──► repositories ──► PostgreSQL
                        │            │
                        │            └──► pure domain (risk/analytics/pipelines)
                        └──► LangGraph assistant ──► grounded tools ──► services
                                                  └──► LLM (OpenAI-compatible | Mock)
```

## Key engineering decisions

| Decision | Choice | Rationale / alternative rejected |
|---|---|---|
| Topology | Modular monolith | One deployable, trivial local dev. Microservices would be cosplay for a single-user analytics tool; the layer boundaries already show how I'd split it (ingestion worker, risk engine). |
| Money type | `NUMERIC(18,6)` → `Decimal` | Exact cents; floats lose precision at scale. |
| Domain purity | No I/O in `risk/`/`analytics/` | Math is unit-testable with no DB — "prove your VaR" is one pytest file. |
| Sync SQLAlchemy | Not async | Workload is CPU-bound (NumPy), not I/O-bound; async adds complexity for no throughput gain here. |
| LLM grounding | Tools + validate node + mock | The model cannot fabricate a metric it has no tool to fetch; runs offline. |
| Frontend | Streamlit | Fastest path to a 7-page analytical dashboard; consumes the API like any client. |

## Scalability considerations

- **Reads scale** with Postgres indexes (`(ticker, date)`, portfolio FKs) and set-based
  queries (`latest_bars` is one grouped-subquery join — no N+1).
- **Ingestion scales** via single-statement `ON CONFLICT` bulk upsert; a background
  worker (Celery/RQ) would offload large imports.
- **Risk compute** is vectorized NumPy; heavy portfolios/time-ranges would move to a
  cache (Redis) keyed by (portfolio, as-of, price-version) and/or a task queue.
- **Millions of rows**: partition `market_data` by ticker/date range, add a columnar
  store (or TimescaleDB) for the time series, precompute daily returns.

## Likely failure modes & handling

| Failure | Handling |
|---|---|
| DB unavailable | `/health/ready` fails; entrypoint waits for DB before serving. |
| Insufficient history | Typed `InsufficientHistoricalData` → 409, never a NaN metric. |
| Missing market data | `MarketDataNotFound` → 404 naming the tickers. |
| Bad uploaded data | Validation quarantines + reports; nothing silently dropped. |
| LLM down / no key | Deterministic mock renderer; assistant still answers, grounded. |
| LLM hallucination | Validate node flags numbers absent from tool outputs. |

## Production hardening (what I'd change)

AuthN/Z (JWT/OAuth) on the API; rate limiting; secrets via a manager (not env files);
per-request DB connection pooling tuned; structured logs shipped to a collector;
migrations gated in CI; the risk-compute-on-GET moved behind POST or cached; Alembic
run as a one-shot job rather than in the API entrypoint for multi-replica deploys.
