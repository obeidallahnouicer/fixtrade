"""
Adapter: Stock price repository.

Implements StockPriceRepository port.
Responsible for fetching historical OHLCV data from the data source.
"""

from datetime import date

from app.domain.trading.entities import StockPrice
from app.domain.trading.ports import StockPriceRepository


class StockPriceRepositoryAdapter(StockPriceRepository):
    """Concrete adapter for retrieving stock price data.

    Implements the StockPriceRepository port defined in the domain layer.
    In production, this will connect to a database or external data API.
    """

    def __init__(self) -> None:
        # TODO: inject database session or API client
        pass

    def get_history(
        self, symbol: str, start: date, end: date
    ) -> list[StockPrice]:
        """Return OHLCV history for a symbol within the date range.

        Args:
            symbol: BVMT stock ticker.
            start: Start date of the range (inclusive).
            end: End date of the range (inclusive).

        Returns:
            List of StockPrice records ordered by date ascending.
        """
        # TODO: query database or external API for price data
        raise NotImplementedError("StockPriceRepositoryAdapter.get_history")
