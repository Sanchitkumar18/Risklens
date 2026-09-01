# RiskLens — Database Design

PostgreSQL via SQLAlchemy 2.0 (typed `Mapped[...]`) and Alembic migrations. The schema
also runs on SQLite, which the in-memory test suite uses (portability via
`with_variant`); production is Postgres.

## Tables

### `market_data` — daily OHLCV bars
`id, ticker, date, open, high, low, close, adjusted_close, volume, created_at`
- `UNIQUE(ticker, date)` → **idempotent ingestion** (re-import updates in place).
- `CHECK(high >= low)`, positive-price check, non-negative volume — bad data can't land.
- `INDEX(ticker, date)` for per-asset range scans.

### `portfolios`
`id, name (UNIQUE), description, created_at` — cascade to positions.

### `positions`
`id, portfolio_id (FK cascade), ticker, quantity, average_price, created_at`
- `UNIQUE(portfolio_id, ticker)` → one row per asset per portfolio.
- `CHECK(quantity <> 0)`; `INDEX(portfolio_id)`.

### `risk_metrics` — **append-only** calculation snapshots
`id, portfolio_id (FK), calculation_date, confidence_level, var, parametric_var,
volatility_annualized, max_drawdown, exposure_gross, exposure_net, stress_loss, method,
created_at`
- Kept as history (not upserted) so the assistant can answer *"why did risk change?"* by
  diffing successive rows. `INDEX(portfolio_id, calculation_date)`.

### `anomalies`
`id, portfolio_id (FK, nullable), ticker, date, anomaly_score, anomaly_type, features
(JSONB), created_at` — `features` is `JSONB` on Postgres (indexable), `JSON` on SQLite.
Indexes on `(ticker, date)` and `portfolio_id`.

### `alerts`
`id, portfolio_id (FK), alert_type, severity, message, metric_value, threshold,
dedup_key (UNIQUE), acknowledged, created_at`
- `UNIQUE(dedup_key)` → **database-enforced de-duplication** (`portfolio:type[:ticker]:date`);
  the engine cannot insert the same logical breach twice.
- `INDEX(portfolio_id, acknowledged)`.

## Design rationale

- **`NUMERIC(18,6)` everywhere for money/prices**, mapped to Python `Decimal`. Binary
  floats can't represent decimal cents exactly — a classic correctness (and interview)
  point.
- **Constraints in the database, not just the app**: uniqueness (ingestion idempotency,
  one-row-per-asset, alert dedup) and CHECKs (OHLC sanity) are the durable backstop.
- **Cross-dialect portability**: PKs are `BigInteger().with_variant(Integer, "sqlite")`
  (SQLite only auto-increments an `INTEGER PRIMARY KEY`); `features` uses a JSONB/JSON
  variant. The identical models + migration run on Postgres and SQLite.
- **Append-only `risk_metrics`** trades storage for auditability and history.
- **Repositories own queries; services own transactions.** Repos `flush` (surface
  constraint violations, assign keys) but never `commit`, so multiple repo calls compose
  into one atomic request.

## Migrations

Hand-written initial migration (`alembic/versions/0001_initial_schema.py`) for explicit
constraint/index names; URL injected from settings (no secrets in `alembic.ini`).
`alembic revision --autogenerate` reports **no drift** against the ORM models.

```bash
make migrate        # alembic upgrade head
make migrate-down   # downgrade one revision
```
