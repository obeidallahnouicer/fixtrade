"""
Centralized error handlers for FastAPI.

Maps domain-specific errors to HTTP responses.
No stack traces or internal details are exposed to clients.
All error responses use the ErrorResponse schema.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.domain.trading.errors import (
    AnomalyDetectionError,
    InsufficientFundsError,
    InsufficientLiquidityError,
    InvalidHorizonError,
    PortfolioNotFoundError,
    SymbolNotFoundError,
    TradingDomainError,
)

logger = logging.getLogger(__name__)

HTTP_400 = 400
HTTP_404 = 404
HTTP_422 = 422
HTTP_500 = 500


def _error_response(status_code: int, error: str, detail: str | None = None) -> JSONResponse:
    """Build a consistent JSON error response."""
    body: dict[str, str | None] = {"error": error}
    if detail:
        body["detail"] = detail
    return JSONResponse(status_code=status_code, content=body)


def register_error_handlers(app: FastAPI) -> None:
    """Register all domain error handlers on the FastAPI application.

    Args:
        app: The FastAPI application instance.
    """

    @app.exception_handler(SymbolNotFoundError)
    async def handle_symbol_not_found(
        _request: Request, exc: SymbolNotFoundError
    ) -> JSONResponse:
        """Handle missing stock symbol errors."""
        logger.warning("Symbol not found: %s", exc.symbol)
        return _error_response(HTTP_404, "Symbol not found")

    @app.exception_handler(InvalidHorizonError)
    async def handle_invalid_horizon(
        _request: Request, exc: InvalidHorizonError
    ) -> JSONResponse:
        """Handle invalid prediction horizon errors."""
        logger.warning("Invalid horizon: %d", exc.horizon)
        return _error_response(HTTP_422, "Invalid prediction horizon")

    @app.exception_handler(InsufficientLiquidityError)
    async def handle_insufficient_liquidity(
        _request: Request, exc: InsufficientLiquidityError
    ) -> JSONResponse:
        """Handle insufficient liquidity errors."""
        logger.warning("Insufficient liquidity: %s", exc.symbol)
        return _error_response(HTTP_400, "Insufficient liquidity")

    @app.exception_handler(InsufficientFundsError)
    async def handle_insufficient_funds(
        _request: Request, exc: InsufficientFundsError
    ) -> JSONResponse:
        """Handle insufficient funds errors."""
        logger.warning("Insufficient funds")
        return _error_response(HTTP_400, "Insufficient funds")

    @app.exception_handler(PortfolioNotFoundError)
    async def handle_portfolio_not_found(
        _request: Request, exc: PortfolioNotFoundError
    ) -> JSONResponse:
        """Handle missing portfolio errors."""
        logger.warning("Portfolio not found: %s", exc.portfolio_id)
        return _error_response(HTTP_404, "Portfolio not found")

    @app.exception_handler(AnomalyDetectionError)
    async def handle_anomaly_detection(
        _request: Request, exc: AnomalyDetectionError
    ) -> JSONResponse:
        """Handle anomaly detection failures."""
        logger.error("Anomaly detection error: %s", exc.reason)
        return _error_response(HTTP_500, "Anomaly detection failed")

    @app.exception_handler(TradingDomainError)
    async def handle_trading_domain(
        _request: Request, exc: TradingDomainError
    ) -> JSONResponse:
        """Catch-all for unhandled trading domain errors."""
        logger.error("Unhandled trading domain error: %s", exc.message)
        return _error_response(HTTP_500, "Internal server error")

    @app.exception_handler(Exception)
    async def handle_unexpected(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all for unexpected errors. Never exposes internals."""
        logger.exception("Unexpected error: %s", type(exc).__name__)
        return _error_response(HTTP_500, "Internal server error")
