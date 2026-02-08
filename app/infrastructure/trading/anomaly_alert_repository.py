"""
Adapter: Anomaly alert repository.

Implements AnomalyAlertRepository port.
Responsible for persisting and retrieving anomaly alerts from PostgreSQL.
"""

import os
from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID

import psycopg2
from psycopg2.extras import RealDictCursor

from app.domain.trading.entities import AnomalyAlert
from app.domain.trading.ports import AnomalyAlertRepository


class AnomalyAlertRepositoryAdapter(AnomalyAlertRepository):
    """PostgreSQL implementation of anomaly alert repository.

    Stores and retrieves anomaly alerts from the anomaly_alerts table.
    """

    def __init__(self) -> None:
        """Initialize repository with database connection parameters."""
        self._db_url = os.getenv("DATABASE_URL")
        if not self._db_url:
            raise ValueError("DATABASE_URL environment variable not set")

    def _get_connection(self):
        """Create a new database connection."""
        return psycopg2.connect(self._db_url)

    def save(self, alert: AnomalyAlert) -> None:
        """Persist a single anomaly alert.

        Args:
            alert: The anomaly alert to save.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO anomaly_alerts
                        (id, symbol, detected_at, anomaly_type, severity, description)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        str(alert.id),
                        alert.symbol,
                        alert.detected_at,
                        alert.anomaly_type,
                        float(alert.severity),
                        alert.description,
                    ),
                )
            conn.commit()

    def save_batch(self, alerts: list[AnomalyAlert]) -> None:
        """Persist multiple anomaly alerts in a single transaction.

        Args:
            alerts: List of anomaly alerts to save.
        """
        if not alerts:
            return

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                values = [
                    (
                        str(alert.id),
                        alert.symbol,
                        alert.detected_at,
                        alert.anomaly_type,
                        float(alert.severity),
                        alert.description,
                    )
                    for alert in alerts
                ]
                cur.executemany(
                    """
                    INSERT INTO anomaly_alerts
                        (id, symbol, detected_at, anomaly_type, severity, description)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    values,
                )
            conn.commit()

    def get_recent(
        self,
        symbol: Optional[str] = None,
        limit: int = 10,
        since: Optional[datetime] = None,
    ) -> list[AnomalyAlert]:
        """Return recent anomaly alerts.

        Args:
            symbol: Optional filter by stock symbol.
            limit: Maximum number of alerts to return.
            since: Optional filter for alerts detected after this datetime.

        Returns:
            List of anomaly alerts ordered by detected_at descending.
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT id, symbol, detected_at, anomaly_type, severity, description
                    FROM anomaly_alerts
                    WHERE 1=1
                """
                params = []

                if symbol:
                    query += " AND symbol = %s"
                    params.append(symbol)

                if since:
                    query += " AND detected_at >= %s"
                    params.append(since)

                query += " ORDER BY detected_at DESC LIMIT %s"
                params.append(limit)

                cur.execute(query, params)
                rows = cur.fetchall()

                return [
                    AnomalyAlert(
                        id=UUID(row["id"]),
                        symbol=row["symbol"],
                        detected_at=row["detected_at"],
                        anomaly_type=row["anomaly_type"],
                        severity=Decimal(str(row["severity"])),
                        description=row["description"],
                    )
                    for row in rows
                ]
