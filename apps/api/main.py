import os
import time
import uuid
from collections import defaultdict
from threading import Lock
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# config must be imported first — load_dotenv() runs at module level
from apps.api.config import load_settings  # noqa: E402
from apps.api.logger import configure_logging, get_logger

from apps.api.api.auth import router as auth_router
from apps.api.api.chat import router as chat_router
from apps.api.api.upload import router as upload_router
from services.rag.embeddings import MODEL_NAME as EMBEDDING_MODEL_NAME

settings = load_settings()
configure_logging(settings.log_level, settings.log_format)
logger = get_logger(__name__)
APP_VERSION = "1.0.0"
APP_START_TIME = time.time()

OPENAPI_TAGS = [
    {
        "name": "auth",
        "description": "Login and JWT token issuance.",
    },
    {
        "name": "chat",
        "description": "Grounded chat answers from retrieved context.",
    },
    {
        "name": "upload",
        "description": "Admin-only document upload and ingestion.",
    },
]


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    def vector_store_status() -> str:
        """Return current vector store status for health checks."""

        try:
            from services.rag import vector_store

            store = getattr(vector_store, "_STORE", None)
            if not store:
                return "unknown"
            if not hasattr(store, "index") or store.index is None:
                return "empty"
            if store.index.ntotal == 0:
                return "empty"
            return "ready"
        except Exception:
            return "unknown"

    app = FastAPI(
        title="AI RAG Chatbot API",
        description=(
            "A retrieval-augmented chatbot with grounded responses and authentication."
        ),
        version="1.0.0",
        openapi_tags=OPENAPI_TAGS,
    )

    environment = os.getenv("ENVIRONMENT", "development")
    logger.info(
        "startup.config_loaded",
        extra={"app_name": settings.app_name, "environment": environment},
    )

    # Warn on missing critical env vars
    if not os.getenv("GEMINI_API_KEY"):
        logger.warning(
            "startup.missing_env",
            extra={"variable": "GEMINI_API_KEY", "impact": "LLM calls will fail"},
        )

    logger.info(
        "startup.models",
        extra={
            "llm_model": "gemini-2.0-flash",
            "llm_provider": "Google Gemini",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "model_initialized": True,
        },
    )
    logger.info(
        "startup.vector_store",
        extra={"vector_store": vector_store_status()},
    )
    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(upload_router)

    @app.get("/health")
    async def health_check() -> dict[str, Any]:
        """Return health status for liveness checks."""

        uptime_s = round(time.time() - APP_START_TIME, 2)
        return {
            "status": "ok",
            "version": APP_VERSION,
            "uptime": f"{uptime_s}s",
            "vector_store": vector_store_status(),
            "llm_provider": "Google Gemini",
            "model": "gemini-2.0-flash",
        }

    # --- Improvement #2: simple in-memory rate limiter ---
    _rate_lock = Lock()
    _rate_map: dict[str, list[float]] = defaultdict(list)
    RATE_LIMIT = 30  # requests per window
    RATE_WINDOW_S = 60  # window in seconds
    _RATE_MAP_MAX = 10_000  # bounded cleanup threshold

    @app.middleware("http")
    async def rate_limit_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Enforce per-IP rate limiting."""

        client_ip = request.client.host if request.client else "unknown"
        request_id = getattr(request.state, "request_id", "-")
        now = time.time()
        with _rate_lock:
            # bounded cleanup: prevent unbounded memory growth
            if len(_rate_map) > _RATE_MAP_MAX:
                _rate_map.clear()
            timestamps = _rate_map[client_ip]
            # prune old entries for this IP
            _rate_map[client_ip] = [t for t in timestamps if now - t < RATE_WINDOW_S]
            current_window_size = len(_rate_map[client_ip])
            if current_window_size >= RATE_LIMIT:
                logger.warning(
                    "rate_limit.exceeded",
                    extra={
                        "request_id": request_id,
                        "client_ip": client_ip,
                        "limit": RATE_LIMIT,
                        "current_window_size": current_window_size,
                    },
                )
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Try again later."},
                )
            _rate_map[client_ip].append(now)
        return await call_next(request)

    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Log incoming HTTP requests with structured summary."""

        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id
        start_time = time.perf_counter()
        logger.info(
            "request.start",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "endpoint": request.url.path,
            },
        )
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request.error",
                extra={
                    "request_id": request_id,
                    "endpoint": request.url.path,
                    "error": "Unhandled exception",
                },
            )
            raise
        duration_ms = (time.perf_counter() - start_time) * 1000

        # --- Improvement #3: structured request summary ---
        logger.info(
            "request.summary",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        response.headers["X-Request-Id"] = request_id
        return response

    # CORS: restrict to known frontend origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()
