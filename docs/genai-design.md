# RiskLens — GenAI Assistant Design

The assistant answers natural-language questions about a portfolio's risk. Its defining
property is **grounding**: every number it states originates from the analytical engine,
never from the language model.

## Why tools instead of letting the LLM compute

An LLM has no market data and cannot reliably compute a 5-year percentile, a covariance
decomposition, or a stress P&L. So the LLM is confined to *selecting tools* and
*explaining their output*. The tools (`app/llm/tools.py`) are thin wrappers over the
**same services** the REST API uses — so an assistant figure is the identical audited
figure the API and dashboard show. This is enforced structurally, not by prompting.

Nine tools: `get_portfolio_summary`, `get_risk_metrics`, `get_asset_exposure`,
`get_correlation_matrix`, `get_drawdown_analysis`, `get_risk_contributions`,
`run_stress_test`, `get_anomalies`, `get_alerts`.

## LangGraph workflow

Typed state (`AssistantState`) flows through a linear graph:

```
question ─► classify ─► plan ─► retrieve ─► explain ─► validate ─► answer
```

1. **classify** — deterministic keyword rules map the question to an intent
   (risk_summary, var_explain, contribution, drawdown, correlation, exposure, stress,
   anomalies, alerts). Deterministic = reproducible, testable, no LLM call to route.
2. **plan** — intent → ordered list of tools (and, for stress, which scenario the
   question implies, e.g. "technology … 25%" → `tech_selloff`).
3. **retrieve** — execute the tools against the grounded toolkit. Each call is guarded:
   a data problem becomes `{"error": ...}` in the results, not a crash.
4. **explain** — build the prose. **Mock mode** (no API key): a deterministic renderer
   writes the answer using only tool numbers. **LLM mode**: the model writes it under a
   strict grounding system prompt with the tool results as context.
5. **validate** — the anti-hallucination backstop: extract every number from the draft
   and flag any that don't match a value in the tool results (ISO dates excluded). In
   mock mode the renderer is grounded by construction; in LLM mode this catches
   fabrication.

## Provider abstraction (works with no API key)

`app/llm/providers.py::get_chat_model` returns a real `ChatOpenAI` (any OpenAI-compatible
endpoint via `OPENAI_BASE_URL`) when `LLM_PROVIDER=openai` and a key is set, otherwise a
deterministic `MockChatModel`. The deterministic explanation renderer means grounding
never depends on a live model — the assistant is fully functional and testable offline.

## Safety & grounding rules

The assistant must not: fabricate metrics or prices, give personalized investment advice,
present simulated data as real, or claim VaR is a maximum loss. If information is
unavailable it says so. Every answer carries a *not financial advice* disclaimer. VaR is
described as a threshold exceeded with probability ≈ (1 − confidence) **based on the
historical sample** — explicitly not a guarantee.

## How hallucination is prevented (summary)

1. The LLM has **no data access** except tools.
2. Tools return **audited service output**, not LLM text.
3. The **validate node** rejects/flags numbers not present in tool results.
4. The **end-to-end test asserts** `assistant.VaR == risk_engine.VaR` — grounding is
   machine-checked, not just designed.

## Example (mock mode, real output)

> **Q:** Explain my 99% VaR in simple terms.
> **A:** The 99% 1-day historical VaR is 3.09% (~$5,715 of a $184,657 portfolio).
> Interpretation: based on 1565 historical daily returns, a one-day loss greater than
> about $5,715 occurred in roughly 1% of days. It is not a maximum — larger losses can
> and do happen beyond this threshold. For comparison, the parametric (Normal) estimate
> is $5,585; it can understate risk when returns have fat tails. This is analytics on
> synthetic/demo data, not financial advice.
