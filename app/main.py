"""FastAPI application factory.

Assembles the ASGI application: configures logging, registers routers, installs the
single domain-exception handler that converts ``RiskLensError`` subclasses into the
standard ``{"error": {...}}`` envelope, and wires request logging middleware.

Run locally with:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import (
    alerts,
    anomalies,
    assistant,
    health,
    market_data,
    portfolios,
    risk,
    stress,
)
from app.core.config import get_settings
from app.core.exceptions import RiskLensError
from app.core.logging import configure_logging, get_logger

API_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks. Logging is configured before the app serves traffic."""
    settings = get_settings()
    configure_logging(level=settings.log_level, json_format=settings.log_json)
    logger = get_logger("risklens.startup")
    logger.info(
        "RiskLens starting",
        extra={"environment": settings.app_env, "llm_provider": settings.llm_provider},
    )
    yield
    logger.info("RiskLens shutting down")


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    settings = get_settings()
    logger = get_logger("risklens.request")

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Market Risk Analytics & GenAI Assistant — research/demo, not financial advice.",
        lifespan=lifespan,
    )

    # ── Request logging + correlation id ────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": elapsed_ms,
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response

    # ── Domain exception handler → structured envelope ──────
    @app.exception_handler(RiskLensError)
    async def handle_domain_error(request: Request, exc: RiskLensError) -> JSONResponse:
        # 5xx errors are logged with traceback server-side; the client sees no stack.
        if exc.http_status >= 500:
            logger.exception("domain error", extra={"code": exc.code, "path": request.url.path})
        else:
            logger.warning(
                "domain error",
                extra={"code": exc.code, "status": exc.http_status, "path": request.url.path},
            )
        return JSONResponse(status_code=exc.http_status, content={"error": exc.to_dict()})

    # ── Pydantic request validation → 422 envelope ──────────
    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "REQUEST_VALIDATION_ERROR",
                    "message": "Request payload failed validation.",
                    "details": {"errors": exc.errors()},
                }
            },
        )

    # ── Routers ─────────────────────────────────────────────
    for module in (health, market_data, portfolios, risk, stress, anomalies, alerts, assistant):
        app.include_router(module.router, prefix=API_PREFIX)

    return app


app = create_app()
