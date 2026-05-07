import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (
            "trace_id",
            "user_id",
            "tenant_id",
            "request_id",
            "method",
            "path",
            "endpoint",
            "status_code",
            "duration_ms",
            "retrieval_time_ms",
            "llm_time_ms",
            "total_time_ms",
            "cache_hit",
            "cache_hit_rate",
            "context_count",
            "query_length",
            "retrieved_count",
            "total_requests",
            "total_errors",
            "vector_store",
            "uptime_s",
            "environment",
            "app_name",
            "llm_model",
            "model",
            "embedding_model",
            "model_initialized",
            "error",
            "client_ip",
            "limit",
            "current_window_size",
            "attempt",
            "variable",
            "impact",
            "document_id",
            "chunks",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if "request_id" not in payload:
            payload["request_id"] = "-"
        if record.exc_info:
            if "error" not in payload:
                payload["error"] = self.formatException(record.exc_info)
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(log_level: str, log_format: str) -> None:
    """Configure root logging handlers and format."""

    level = getattr(logging, log_level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    if log_format.lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger instance."""

    return logging.getLogger(name)
