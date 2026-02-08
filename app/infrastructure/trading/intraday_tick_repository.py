"""
Adapter: Intraday tick repository.

Implements IntradayTickRepository port.
Reads/writes 1-minute bars and tick-by-tick data from PostgreSQL.
"""

import os
from datetime import datetime
from decimal import Decimal

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from app.domain.trading.entities import IntradayTick
from app.domain.trading.ports import IntradayTickRepository


class IntradayTickRepositoryAdapter(IntradayTickRepository):
    """Concrete adapter for intraday tick data.

    Implements the IntradayTickRepository port defined in the domain layer.
    Connects to PostgreSQL to read/write intraday_ticks table.
    """

    def __init__(self) -> None:
        """Initialize with database connection parameters."""
        self._db_url = os.getenv("DATABASE_URL")
        if not self._db_url:
            raise ValueError("DATABASE_URL environment variable not set")

    def _get_connection(self):
        """Create a new database connection."""
        return psycopg2.connect(self._db_url)

    def get_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        tick_type: str = "1min",
    ) -> list[IntradayTick]:
        """Return intraday ticks for a symbol within the datetime range.

        Args:
            symbol: BVMT stock ticker.
            start: Start datetime (inclusive).
            end: End datetime (inclusive).
            tick_type: Filter by tick type ("1min" or "tick").

        Returns:
            List of IntradayTick ordered by timestamp ascending.
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT symbol, tick_timestamp, price, volume, tick_type
                    FROM intraday_ticks
                    WHERE symbol = %s
                      AND tick_timestamp >= %s
                      AND tick_timestamp <= %s
                      AND tick_type = %s
                    ORDER BY tick_timestamp ASC
                    """,
                    (symbol, start, end, tick_type),
                )
                rows = cur.fetchall()

                return [
                    IntradayTick(
                        symbol=row["symbol"],
                        timestamp=row["tick_timestamp"],
                        price=Decimal(str(row["price"])),
                        volume=row["volume"] or 0,
                        tick_type=row["tick_type"],
                    )
                    for row in rows
                ]

    def save_batch(self, ticks: list[IntradayTick]) -> int:
        """Persist a batch of intraday ticks.

        Uses ON CONFLICT DO NOTHING for idempotent inserts.

        Args:
            ticks: List of IntradayTick entities.

        Returns:
            Number of rows inserted.
        """
        if not ticks:
            return 0

        values = [
            (t.symbol, t.timestamp, float(t.price), t.volume, t.tick_type)
            for t in ticks
        ]

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO intraday_ticks
                        (symbol, tick_timestamp, price, volume, tick_type)
                    VALUES %s
                    ON CONFLICT (symbol, tick_timestamp, tick_type) DO NOTHING
                    """,
                    values,
                    page_size=5000,
                )
                inserted = cur.rowcount
            conn.commit()

        return inserted

    def get_symbols_with_data(self, since: datetime) -> list[str]:
        """Return symbols that have intraday data since a given datetime."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT symbol
                    FROM intraday_ticks
                    WHERE tick_timestamp >= %s
                    ORDER BY symbol
                    """,
                    (since,),
                )
                return [row[0] for row in cur.fetchall()]
