"""
Domain-specific errors for the trading bounded context.

All errors raised from the domain layer must be defined here.
These are mapped to HTTP responses at the interface layer.
No framework imports allowed.
"""


class TradingDomainError(Exception):
    """Base error for all trading domain errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class SymbolNotFoundError(TradingDomainError):
    """Raised when a stock symbol is not found in the BVMT catalog."""

    def __init__(self, symbol: str) -> None:
        super().__init__(f"Symbol not found: {symbol}")
        self.symbol = symbol


class InsufficientLiquidityError(TradingDomainError):
    """Raised when there is not enough liquidity to execute a trade."""

    def __init__(self, symbol: str) -> None:
        super().__init__(f"Insufficient liquidity for symbol: {symbol}")
        self.symbol = symbol


class InsufficientFundsError(TradingDomainError):
    """Raised when the portfolio lacks funds for a purchase."""

    def __init__(self, required: str, available: str) -> None:
        super().__init__(
            f"Insufficient funds: required {required}, available {available}"
        )
        self.required = required
        self.available = available


class InvalidHorizonError(TradingDomainError):
    """Raised when the prediction horizon is outside the allowed range."""

    def __init__(self, horizon: int) -> None:
        super().__init__(
            f"Invalid prediction horizon: {horizon}. Must be between 1 and 5."
        )
        self.horizon = horizon


class PortfolioNotFoundError(TradingDomainError):
    """Raised when a portfolio cannot be found."""

    def __init__(self, portfolio_id: str) -> None:
        super().__init__(f"Portfolio not found: {portfolio_id}")
        self.portfolio_id = portfolio_id


class AnomalyDetectionError(TradingDomainError):
    """Raised when the anomaly detection process fails."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Anomaly detection failed: {reason}")
        self.reason = reason
