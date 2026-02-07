from fastapi import FastAPI
from app.api.v1.router import router as api_router
from app.core.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title=settings.project_name)
    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
