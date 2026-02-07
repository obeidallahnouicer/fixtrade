"""
Application entry point.

Creates the FastAPI application and wires together:
- Routers (one per bounded context)
- Error handlers (centralized domain-to-HTTP mapping)
- Security middleware (headers, rate limiting)
- Logging configuration

No business logic belongs here.
"""

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
