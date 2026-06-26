"""Portfolio optimization HTTP endpoints."""

from fastapi import APIRouter, HTTPException

from app.interfaces.portfolio.schemas import (
    PortfolioOptimizationRequest,
    PortfolioOptimizationResponse,
)
from app.interfaces.portfolio.service import optimize_portfolio

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.post(
    "/optimize",
    response_model=PortfolioOptimizationResponse,
    summary="Build a CAPM minimum-variance BVMT portfolio",
)
def build_portfolio(
    payload: PortfolioOptimizationRequest,
) -> PortfolioOptimizationResponse:
    try:
        return optimize_portfolio(payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="La construction du portefeuille est temporairement indisponible.",
        ) from exc
