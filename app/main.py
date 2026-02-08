"""
Application entry point.

Creates the FastAPI application and wires together:
- Routers (one per bounded context)
- Error handlers (centralized domain-to-HTTP mapping)
- Security middleware (headers, rate limiting)
- Logging configuration
- Real-time pipeline (WebSocket, scheduler, file watcher)

No business logic belongs here.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.interfaces.health import router as health_router
from app.interfaces.trading.router import router as trading_router
from app.shared.errors.handlers import register_error_handlers
from app.shared.logging import configure_logging
from app.shared.security.headers import SecurityHeadersMiddleware
from app.shared.security.rate_limiting import limiter


# ── Real-time components (optional, degrade gracefully) ──────────
_stream_manager = None
_scheduler = None
_watcher = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: start/stop real-time pipeline components."""
    global _stream_manager, _scheduler, _watcher

    try:
        from prediction.realtime.stream import PredictionStreamManager
        from prediction.realtime.scheduler import RealtimeScheduler
        from prediction.realtime.watcher import DataWatcher
        from app.interfaces.realtime import (
            router as realtime_router,
            set_realtime_components,
        )

        _stream_manager = PredictionStreamManager()
        _scheduler = RealtimeScheduler(
            stream_manager=_stream_manager,
            top_n_tickers=10,
        )
        _watcher = DataWatcher(
            poll_interval=30.0,
            auto_retrain=False,
            scheduler=_scheduler,
            stream_manager=_stream_manager,
        )

        set_realtime_components(_stream_manager, _scheduler, _watcher)

        # Start scheduler and watcher
        _scheduler.start()
        _watcher.start()

        # Mount the realtime router dynamically
        app.include_router(realtime_router, prefix="/api/v1")

    except Exception:
        import logging
        logging.getLogger(__name__).warning(
            "Real-time pipeline components could not be initialized. "
            "WebSocket/scheduler/watcher features are disabled.",
            exc_info=True,
        )

    yield

    # Shutdown
    if _scheduler is not None:
        _scheduler.stop()
    if _watcher is not None:
        _watcher.stop()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Registers routers, error handlers, and security middleware.
    This is the composition root of the application.

    Returns:
        A fully configured FastAPI application instance.
    """
    configure_logging(level=settings.log_level)

    app = FastAPI(
        title=settings.project_name,
        version=settings.version,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    # --- Rate Limiting ---
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # --- Security Middleware ---
    app.add_middleware(SecurityHeadersMiddleware)

    # --- Error Handlers ---
    register_error_handlers(app)

    # --- Routers ---
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(trading_router, prefix="/api/v1")

    return app


app = create_app()
