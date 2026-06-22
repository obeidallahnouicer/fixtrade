from fastapi import FastAPI

from app.interfaces.auth import router as auth_router
from app.interfaces.health import router as health_router


def create_app() -> FastAPI:
    app = FastAPI(title="FixTrade Auth Service", version="1.0.0")
    # Auth endpoints
    app.include_router(auth_router.router, prefix="/api/v1/auth")
    # Lightweight health endpoint for readiness/liveness probes
    app.include_router(health_router, prefix="/api/v1")
    return app


app = create_app()
