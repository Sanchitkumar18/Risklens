"""Structured application logging.

Provides a single ``configure_logging()`` entrypoint (called once on app startup)
and a ``get_logger()`` helper. Two formats are supported:

* human-readable (default, for local development)
* JSON (``LOG_JSON=true``, for production log aggregation)

Secrets are never logged: this module only formats records it is given. Call sites
are responsible for not passing secrets — see ``core/config.py`` which never logs
``OPENAI_API_KEY`` or ``DATABASE_URL`` credentials.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

_CONFIGURED = False

# Standard LogRecord attributes we do NOT want to duplicate into the JSON `extra` blob.
_RESERVED_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects for machine ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Merge any structured `extra={...}` fields passed at the call site.
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure the root logger. Idempotent — safe to call more than once.

    Args:
        level: Minimum log level name (e.g. ``"INFO"``).
        json_format: When True, emit JSON lines; otherwise a readable text format.
    """
    global _CONFIGURED

    root = logging.getLogger()
    root.setLevel(level.upper())

    # Reset handlers so re-configuration (e.g. tests) doesn't duplicate output.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root.addHandler(handler)

    # Tame noisy third-party loggers.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring logging with defaults if needed."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
