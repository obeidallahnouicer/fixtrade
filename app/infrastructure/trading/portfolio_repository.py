"""
Adapter: Portfolio persistence.

Implements PortfolioRepository port.
Responsible for persisting and retrieving portfolio data.
"""

from typing import Optional
from uuid import UUID

from app.domain.trading.entities import Portfolio
from app.domain.trading.ports import PortfolioRepository


class PortfolioRepositoryAdapter(PortfolioRepository):
    """Concrete adapter for portfolio data persistence.

    Implements the PortfolioRepository port defined in the domain layer.
    In production, this will connect to a database.
    """

    def __init__(self) -> None:
        # TODO: inject database session
        pass

    def get_by_id(self, portfolio_id: UUID) -> Optional[Portfolio]:
        """Return a portfolio by its ID, or None if not found.

        Args:
            portfolio_id: UUID of the portfolio to retrieve.

        Returns:
            Portfolio entity or None.
        """
        # TODO: query database for portfolio
        raise NotImplementedError("PortfolioRepositoryAdapter.get_by_id")

    def save(self, portfolio: Portfolio) -> None:
        """Persist a portfolio.

        Args:
            portfolio: Portfolio entity to save.
        """
        # TODO: persist portfolio to database
        raise NotImplementedError("PortfolioRepositoryAdapter.save")
