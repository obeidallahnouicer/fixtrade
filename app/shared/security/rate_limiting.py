"""
Rate limiting configuration and setup.

Uses slowapi to enforce per-endpoint rate limits.
Protects against denial-of-service and resource abuse.
"""

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request
from starlette.responses import JSONResponse

DEFAULT_RATE_LIMIT = "60/minute"
HEAVY_RATE_LIMIT = "10/minute"

limiter = Limiter(key_func=get_remote_address, default_limits=[DEFAULT_RATE_LIMIT])


async def rate_limit_exceeded_handler(
    _request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Handle rate limit exceeded errors with a clean JSON response.

    Args:
        _request: The incoming HTTP request.
        exc: The rate limit exceeded exception.

    Returns:
        A 429 JSON response with a clear error message.
    """
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded", "detail": str(exc.detail)},
    )
