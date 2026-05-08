import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
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
from apps.api.api.chats import router as chats_router
from apps.api.api.chat_messages import router as chat_messages_router
from apps.api.api.chat_upload import router as chat_upload_router
from services.rag.embeddings import MODEL_NAME as EMBEDDING_MODEL_NAME

settings = load_settings()
configure_logging(settings.log_level, settings.log_format)
logger = get_logger(__name__)
APP_VERSION = "2.0.0"
APP_START_TIME = time.time()

OPENAPI_TAGS = [
    {
        "name": "auth",
        "description": "Login and JWT token issuance.",
    },
    {
        "name": "chat",
        "description": "Legacy grounded chat endpoint (backwards compatible).",
    },
    {
        "name": "chats",
        "description": "Multi-chat session management (create, list, delete, rename).",
    },
    {
        "name": "messages",
        "description": "Chat-scoped message history and sending.",
    },
    {
        "name": "documents",
        "description": "Chat-scoped document upload and management.",
    },
    {
        "name": "upload",
        "description": "Legacy admin-only document upload (backwards compatible).",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize DB on startup."""
    from services.db.database import init_db
    init_db()
    logger.info("startup.db_initialized")
    yield
    logger.info("shutdown.complete")


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


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="AI RAG Chatbot API",
        description=(
            "A retrieval-augmented chatbot with grounded responses, "
            "persistent chat history, and file-aware multi-chat support."
        ),
        version=APP_VERSION,
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
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
            "llm_model": "gemini-1.5-flash",
            "llm_provider": "Google Gemini",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "model_initialized": True,
        },
    )

    # Register all routers
    app.include_router(auth_router)
    app.include_router(chat_router)          # legacy /chat
    app.include_router(upload_router)        # legacy /upload
    app.include_router(chats_router)         # /chats CRUD
    app.include_router(chat_messages_router) # /chats/{id}/messages
    app.include_router(chat_upload_router)   # /chats/{id}/upload, /documents/{id}

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
            "model": "gemini-1.5-flash",
        }

    # --- Simple in-memory rate limiter ---
    _rate_lock = Lock()
    _rate_map: dict[str, list[float]] = defaultdict(list)
    RATE_LIMIT_LOCALHOST = 200  # generous limit for local dev
    RATE_LIMIT_DEFAULT = 60    # requests per window for remote IPs
    RATE_WINDOW_S = 60  # window in seconds
    _RATE_MAP_MAX = 10_000  # bounded cleanup threshold

    @app.middleware("http")
    async def rate_limit_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Enforce per-IP rate limiting. OPTIONS preflight is exempt."""

        # Skip rate limiting for CORS preflight requests
        if request.method == "OPTIONS":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        request_id = getattr(request.state, "request_id", "-")
        now = time.time()
        # Localhost gets a higher limit to avoid false positives during dev
        is_local = client_ip in ("127.0.0.1", "::1", "localhost")
        rate_limit = RATE_LIMIT_LOCALHOST if is_local else RATE_LIMIT_DEFAULT
        with _rate_lock:
            if len(_rate_map) > _RATE_MAP_MAX:
                _rate_map.clear()
            timestamps = _rate_map[client_ip]
            _rate_map[client_ip] = [t for t in timestamps if now - t < RATE_WINDOW_S]
            current_window_size = len(_rate_map[client_ip])
            if current_window_size >= rate_limit:
                logger.warning(
                    "rate_limit.exceeded",
                    extra={
                        "request_id": request_id,
                        "client_ip": client_ip,
                        "limit": rate_limit,
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
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()
