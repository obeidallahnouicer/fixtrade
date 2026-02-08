"""
Adapter: Stock price repository.

Implements StockPriceRepository port.
Responsible for fetching historical OHLCV data from the data source.
"""

import os
from datetime import date
from decimal import Decimal

import psycopg2
from psycopg2.extras import RealDictCursor

from app.domain.trading.entities import StockPrice
from app.domain.trading.ports import StockPriceRepository


class StockPriceRepositoryAdapter(StockPriceRepository):
    """Concrete adapter for retrieving stock price data.

    Implements the StockPriceRepository port defined in the domain layer.
    Connects to PostgreSQL database to fetch OHLCV data.
    """

    def __init__(self) -> None:
        """Initialize repository with database connection parameters."""
        self._db_url = os.getenv("DATABASE_URL")
        if not self._db_url:
            raise ValueError("DATABASE_URL environment variable not set")

    def _get_connection(self):
        """Create a new database connection."""
        return psycopg2.connect(self._db_url)

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
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        symbol,
                        seance AS date,
                        ouverture AS open,
                        plus_haut AS high,
                        plus_bas AS low,
                        cloture AS close,
                        quantite_negociee AS volume
                    FROM stock_prices
                    WHERE symbol = %s
                      AND seance >= %s
                      AND seance <= %s
                    ORDER BY seance ASC
                    """,
                    (symbol, start, end),
                )
                rows = cur.fetchall()

                return [
                    StockPrice(
                        symbol=row["symbol"],
                        date=row["date"],
                        open=Decimal(str(row["open"])) if row["open"] else Decimal("0"),
                        high=Decimal(str(row["high"])) if row["high"] else Decimal("0"),
                        low=Decimal(str(row["low"])) if row["low"] else Decimal("0"),
                        close=Decimal(str(row["close"])),
                        volume=row["volume"] or 0,
                    )
                    for row in rows
                ]

