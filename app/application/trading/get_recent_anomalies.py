"""
Use case: Retrieve recent market anomaly alerts.

Input: GetRecentAnomaliesQuery (optional symbol, limit)
Output: list[AnomalyResult]
Side effects: None (read-only query).
Failure cases: None.
"""

import logging
from datetime import datetime, timedelta

from app.application.trading.dtos import AnomalyResult, GetRecentAnomaliesQuery
from app.domain.trading.ports import AnomalyAlertRepository

logger = logging.getLogger(__name__)


class GetRecentAnomaliesUseCase:
    """Orchestrates retrieving recent anomaly alerts.

    This is a read-only query use case that fetches
    recent anomaly alerts from the repository.
    """

    def __init__(self, alert_repo: AnomalyAlertRepository) -> None:
        """Initialize the use case.

        Args:
            alert_repo: Repository for retrieving anomaly alerts.
        """
        self._alert_repo = alert_repo

    def execute(self, query: GetRecentAnomaliesQuery) -> list[AnomalyResult]:
        """Run the get recent anomalies use case.

        Args:
            query: Query parameters (optional symbol, limit, hours_back).

        Returns:
            A list of recent anomaly alerts.
        """
        logger.info(
            "Retrieving recent anomalies: symbol=%s, limit=%d, hours_back=%d",
            query.symbol,
            query.limit,
            query.hours_back,
        )

        # Calculate time threshold
        since = datetime.now() - timedelta(hours=query.hours_back)

        # Fetch alerts from repository
        alerts = self._alert_repo.get_recent(
            symbol=query.symbol,
            limit=query.limit,
            since=since,
        )

        # Map to DTOs
        return [
            AnomalyResult(
                id=alert.id,
                symbol=alert.symbol,
                detected_at=alert.detected_at,
                anomaly_type=alert.anomaly_type,
                severity=alert.severity,
                description=alert.description,
            )
            for alert in alerts
        ]
