"""Thin HTTP client the Streamlit dashboard uses to talk to the RiskLens API.

Keeps all network concerns in one place so views stay declarative. Errors from the
API's ``{"error": {...}}`` envelope are unwrapped into a readable ``RiskLensAPIError``.
"""

from __future__ import annotations

from typing import Any

import httpx


class RiskLensAPIError(RuntimeError):
    """Raised when the API returns an error response."""


class RiskLensClient:
    """Minimal client over the RiskLens REST API."""

    def __init__(self, base_url: str = "http://localhost:8000/api/v1", client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(base_url=self.base_url, timeout=60.0)

    # ── low level ───────────────────────────────────────────
    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            resp = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:  # connection refused, timeout, …
            raise RiskLensAPIError(f"Could not reach the API: {exc}") from exc
        if resp.status_code >= 400:
            raise RiskLensAPIError(self._error_message(resp))
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    @staticmethod
    def _error_message(resp: httpx.Response) -> str:
        try:
            body = resp.json()
            err = body.get("error", {})
            return f"{err.get('code', resp.status_code)}: {err.get('message', resp.text)}"
        except Exception:
            return f"HTTP {resp.status_code}: {resp.text[:200]}"

    # ── health ──────────────────────────────────────────────
    def health(self) -> dict:
        return self._request("GET", "/health")

    # ── market data ─────────────────────────────────────────
    def upload_csv(self, filename: str, content: bytes) -> dict:
        return self._request(
            "POST", "/market-data/upload", files={"file": (filename, content, "text/csv")}
        )

    def get_series(self, ticker: str) -> dict:
        return self._request("GET", f"/market-data/{ticker}")

    # ── portfolios ──────────────────────────────────────────
    def list_portfolios(self) -> list[dict]:
        return self._request("GET", "/portfolios")

    def create_portfolio(self, name: str, description: str | None = None) -> dict:
        return self._request("POST", "/portfolios", json={"name": name, "description": description})

    def get_portfolio(self, pid: int) -> dict:
        return self._request("GET", f"/portfolios/{pid}")

    def add_position(self, pid: int, ticker: str, quantity: float, average_price: float) -> dict:
        return self._request(
            "POST", f"/portfolios/{pid}/positions",
            json={"ticker": ticker, "quantity": str(quantity), "average_price": str(average_price)},
        )

    def remove_position(self, pid: int, position_id: int) -> None:
        return self._request("DELETE", f"/portfolios/{pid}/positions/{position_id}")

    def valuation(self, pid: int) -> dict:
        return self._request("GET", f"/portfolios/{pid}/valuation")

    # ── risk ────────────────────────────────────────────────
    def risk(self, pid: int, confidence: float | None = None) -> dict:
        params = {"confidence": confidence} if confidence else None
        return self._request("GET", f"/portfolios/{pid}/risk", params=params)

    def correlation(self, pid: int) -> dict:
        return self._request("GET", f"/portfolios/{pid}/correlation")

    def risk_timeseries(self, pid: int, vol_window: int = 21) -> list[dict]:
        return self._request(
            "GET", f"/portfolios/{pid}/risk/timeseries", params={"vol_window": vol_window}
        )

    # ── stress ──────────────────────────────────────────────
    def stress_scenarios(self, pid: int) -> list[dict]:
        return self._request("GET", f"/portfolios/{pid}/stress-test/scenarios")

    def stress(self, pid: int, scenario: str) -> dict:
        return self._request("GET", f"/portfolios/{pid}/stress-test", params={"scenario": scenario})

    def stress_custom(self, pid: int, payload: dict) -> dict:
        return self._request("POST", f"/portfolios/{pid}/stress-test/custom", json=payload)

    # ── anomalies ───────────────────────────────────────────
    def anomalies(self, pid: int) -> list[dict]:
        return self._request("GET", f"/portfolios/{pid}/anomalies")

    def scan_anomalies(self, pid: int, contamination: float = 0.02) -> dict:
        return self._request(
            "POST", f"/portfolios/{pid}/anomalies/scan", params={"contamination": contamination}
        )

    # ── alerts ──────────────────────────────────────────────
    def alerts(self, pid: int, acknowledged: bool | None = None) -> list[dict]:
        params = {"acknowledged": acknowledged} if acknowledged is not None else None
        return self._request("GET", f"/portfolios/{pid}/alerts", params=params)

    def scan_alerts(self, pid: int, thresholds: dict | None = None) -> dict:
        return self._request("POST", f"/portfolios/{pid}/alerts/scan", json=thresholds or {})

    def acknowledge_alert(self, pid: int, alert_id: int) -> dict:
        return self._request("POST", f"/portfolios/{pid}/alerts/{alert_id}/acknowledge")

    # ── assistant (Phase 12/13) ─────────────────────────────
    def assistant_query(self, pid: int, question: str) -> dict:
        return self._request(
            "POST", "/assistant/query", json={"portfolio_id": pid, "question": question}
        )
