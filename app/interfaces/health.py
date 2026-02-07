"""
Health check router.

Provides a simple health endpoint for liveness/readiness probes.
No business logic. Returns application status and version.
"""

from fastapi import APIRouter

from app.core.config import settings
from app.interfaces.trading.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns application health status and version.",
)
def health_check() -> HealthResponse:
    """Return current application health status."""
    return HealthResponse(status="ok", version=settings.version)
