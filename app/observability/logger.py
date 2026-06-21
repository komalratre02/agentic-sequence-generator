"""
Structured application logger.

Uses Python's standard logging with rich formatting for local dev,
and JSON-structured output for production.
"""
import logging
import sys
import json
from datetime import datetime, timezone
from app.config import get_settings

settings = get_settings()


class JSONFormatter(logging.Formatter):
    """Emit logs as JSON lines for log-aggregation pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logging() -> None:
    """Configure root logger once at startup."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)

    if settings.app_env == "production":
        handler.setFormatter(JSONFormatter())
    else:
        # Human-readable for local development
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "google.auth", "urllib3.connectionpool"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
