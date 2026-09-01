# RiskLens — Interview Guide

Talking points and defensible answers for the questions this project invites. Numbers
below come from the seeded demo (`make demo`).

---

## Architecture & stack

**Why FastAPI?** Async-capable ASGI, first-class Pydantic validation, and automatic
OpenAPI/Swagger docs — I get typed request/response models, correct HTTP status codes,
and interactive docs essentially for free. Alternatives: Flask (no built-in validation
or schema), Django (heavyweight for an API-first analytics service).

**Why PostgreSQL?** Strong typing including `NUMERIC` (exact decimal money), real
constraints (unique, check, FK cascade) that enforce invariants at the data layer, rich
indexing, `JSONB` for anomaly feature blobs, and a clear path to partitioning/Timescale
for large time series. SQLite is used only for the fast, dependency-free test suite; the
models are written to run on both.

**Why a modular monolith, not microservices?** One deployable, trivial local dev, and no
network boundaries to debug for a single-user analytics tool. The layer boundaries
(pure domain / services / repositories) already document how I'd carve out an ingestion
worker or a risk-engine service if scale demanded it.

**Why sync SQLAlchemy?** The bottleneck is CPU-bound NumPy math, not I/O concurrency.
Async would add complexity (async sessions, greenlets) for no throughput win here.

---

## Risk methodology

**Why historical VaR (as primary)?** It makes no distributional assumption — it reads the
loss directly from the empirical return distribution, so it captures fat tails and the
crash days present in the data. Parametric VaR is implemented alongside for teaching, and
the demo *shows* it understating the tail (99%: parametric $5,585 vs historical $5,715).

**How does VaR work?** *α*-confidence 1-day VaR is the loss threshold exceeded with
probability (1−α). Historical: `VaR = −Quantile_{1−α}(returns) × value`. We report it as a
**positive loss**, consistently. Crucially it is **not a maximum** — losses beyond VaR
occur ~(1−α) of the time.

**Why 95% vs 99%?** They answer different questions: 95% = "a normal bad day" (exceeded
~1 day in 20); 99% = "a rare bad day" (~1 in 100). 99% probes deeper into the tail, so
it's larger ($3,731 vs $5,715 here) and needs more data to estimate stably.

**How would you backtest VaR?** Count exceptions: over N days, how often did the realized
loss exceed the VaR? For 95% you expect ~5%. Then a **Kupiec POF test** (likelihood-ratio
on the exception rate) checks if the breach frequency is statistically consistent with the
confidence level; **Christoffersen** adds an independence test (breaches shouldn't cluster).
That's the natural next feature.

**Risk contribution — why not just each asset's volatility?** Standalone vol ignores
correlation. I use the **Euler decomposition of portfolio volatility**: `σ_p = √(wᵀΣw)`,
`CCR_i = w_i·(Σw)_i/σ_p`, and `Σ CCR_i = σ_p` exactly. In the demo NVDA is 39% of capital
but **55% of risk** — that gap is the whole point, and per-asset vol can't show it.

---

## Machine learning

**Why Isolation Forest?** Anomalies are unlabeled, rare, and multivariate. Isolation
Forest isolates points with random splits — outliers need fewer splits (shorter path →
higher score). It's unsupervised (no labeled crashes), near-linear/scalable,
distribution-free (unlike a z-score/Mahalanobis rule), and genuinely multivariate (it
flags an odd *combination*, e.g. a small price move on freakish volume). Contamination is
the tunable sensitivity; it's a prior on anomaly frequency, not ground truth — a known
limitation.

---

## GenAI

**How does LangGraph work here?** A `StateGraph` over a typed `AssistantState` with nodes
`classify → plan → retrieve → explain → validate`. Each node reads/writes state; edges are
linear. Classification and planning are deterministic (reproducible routing, no LLM call);
the model only writes the final prose.

**Why tools instead of letting the LLM calculate metrics?** The LLM has no market data and
can't reliably compute a 5-year percentile or a covariance decomposition. Tools wrap the
**same services** the API uses, so the assistant's numbers are the identical audited
numbers — the LLM selects tools and explains their output.

**How do you prevent hallucinations?** Four layers: (1) the LLM has no data access except
tools; (2) tools return audited service output, not model text; (3) a **validate node**
extracts every number from the answer and flags any not present in the tool results; (4)
the **end-to-end test asserts `assistant.VaR == engine.VaR`** — grounding is machine-checked.
It also runs offline with a deterministic renderer, so grounding never depends on a live
model.

**Handling model failures?** The provider abstraction falls back to the deterministic
renderer; each tool call is guarded so a data error becomes a structured message ("I don't
have enough data") rather than a crash or a fabricated answer; timeouts/retries would wrap
the real provider in production.

---

## Scale & operations

**Scale to millions of market-data rows?** Partition `market_data` by ticker and/or date
range; consider TimescaleDB/columnar storage for the time series; precompute daily returns;
keep the set-based, index-backed queries (the `latest_bars` grouped-subquery join avoids
N+1); move large imports to a background worker; cache computed risk keyed by
(portfolio, as-of, price-version).

**Make the risk engine real-time?** Stream prices (Kafka), maintain incremental
covariance/EWMA volatility, recompute portfolio VaR on tick with cached factor exposures,
and push updates over WebSockets. Full historical VaR is a periodic (e.g. EOD) job; intraday
uses parametric/EWMA for speed.

**Secure the API?** JWT/OAuth2 auth with per-portfolio authorization; rate limiting; input
validation (already via Pydantic); secrets via a manager (not env files); TLS; no stack
traces to clients (already — single exception handler); audit logging; CORS locked down.

**Deploy to AWS?** Containers on ECS Fargate (API + dashboard) behind an ALB; **RDS
Postgres** (Multi-AZ) as the store; migrations as a one-shot ECS task (not in the API
entrypoint) so multiple replicas don't race; secrets in Secrets Manager; images in ECR;
CloudWatch for logs/metrics; optional Bedrock/OpenAI for the LLM; S3 for raw data uploads;
autoscaling on CPU.

**What would you change for production?** Auth/z + rate limiting; move risk-compute-on-GET
behind POST or a cache; migrations as a separate job; a background queue for ingestion and
heavy compute; VaR backtesting (Kupiec/Christoffersen) surfaced in the UI; real dividend/
split handling so `adjusted_close ≠ close`; observability (tracing, metrics); pinned,
scanned dependencies in CI.

---

## Known limitations (be upfront)

- **Synthetic data** (labeled everywhere) — realistic characteristics, but not real prices.
- `adjusted_close == close` (no dividends/splits modeled).
- VaR/vol assume i.i.d.-ish returns (square-root-of-time scaling); no VaR backtest yet.
- Fixed-contamination anomaly detection flags ~that fraction regardless.
- "Why did risk *change*?" is answered as current drivers; a true snapshot diff is a
  designed-in future feature (the append-only `risk_metrics` table already supports it).
