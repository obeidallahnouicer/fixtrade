"""
Adapter: Known anomalies repository.

Reads labeled/curated anomalies from the known_anomalies table
for use in evaluation and backtesting.
"""

import os
from datetime import date
from decimal import Decimal

import psycopg2
from psycopg2.extras import RealDictCursor

from app.domain.trading.anomaly_evaluator import LabeledAnomaly


class KnownAnomalyRepositoryAdapter:
    """Repository for reading labeled anomalies from PostgreSQL.

    These anomalies are hand-curated ground-truth data used for
    evaluating the anomaly detection service's accuracy.
    """

    def __init__(self) -> None:
        """Initialize with database connection parameters."""
        self._db_url = os.getenv("DATABASE_URL")
        if not self._db_url:
            raise ValueError("DATABASE_URL environment variable not set")

    def _get_connection(self):
        """Create a new database connection."""
        return psycopg2.connect(self._db_url)

    def get_all(
        self,
        symbol: str | None = None,
        verified_only: bool = True,
    ) -> list[LabeledAnomaly]:
        """Return all known anomalies, optionally filtered by symbol.

        Args:
            symbol: Optional filter by stock ticker.
            verified_only: If True, only return verified anomalies.

        Returns:
            List of LabeledAnomaly sorted by date ascending.
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                conditions = []
                params: list = []

                if symbol is not None:
                    conditions.append("symbol = %s")
                    params.append(symbol)

                if verified_only:
                    conditions.append("verified = TRUE")

                where = ""
                if conditions:
                    where = "WHERE " + " AND ".join(conditions)

                cur.execute(
                    f"""
                    SELECT symbol, anomaly_date, anomaly_type, severity,
                           description, source
                    FROM known_anomalies
                    {where}
                    ORDER BY anomaly_date ASC
                    """,
                    params,
                )
                rows = cur.fetchall()

                return [
                    LabeledAnomaly(
                        date=row["anomaly_date"],
                        anomaly_type=row["anomaly_type"],
                        symbol=row["symbol"],
                        description=row.get("description", ""),
                    )
                    for row in rows
                ]

    def get_by_date_range(
        self,
        start: date,
        end: date,
        symbol: str | None = None,
    ) -> list[LabeledAnomaly]:
        """Return known anomalies within a date range.

        Args:
            start: Start date (inclusive).
            end: End date (inclusive).
            symbol: Optional filter by ticker.

        Returns:
            List of LabeledAnomaly sorted by date ascending.
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT symbol, anomaly_date, anomaly_type, severity,
                           description, source
                    FROM known_anomalies
                    WHERE anomaly_date >= %s AND anomaly_date <= %s
                      AND verified = TRUE
                """
                params: list = [start, end]

                if symbol:
                    query += " AND symbol = %s"
                    params.append(symbol)

                query += " ORDER BY anomaly_date ASC"

                cur.execute(query, params)
                rows = cur.fetchall()

                return [
                    LabeledAnomaly(
                        date=row["anomaly_date"],
                        anomaly_type=row["anomaly_type"],
                        symbol=row["symbol"],
                        description=row.get("description", ""),
                    )
                    for row in rows
                ]

    def count(self, symbol: str | None = None) -> int:
        """Return the total number of known anomalies."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if symbol:
                    cur.execute(
                        "SELECT COUNT(*) FROM known_anomalies WHERE symbol = %s",
                        (symbol,),
                    )
                else:
                    cur.execute("SELECT COUNT(*) FROM known_anomalies")
                return cur.fetchone()[0]
