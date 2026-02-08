"""
Use case: Detect market anomalies for a BVMT-listed symbol.

Input: DetectAnomaliesQuery (symbol)
Output: list[AnomalyResult]
Side effects: None.
Failure cases: SymbolNotFoundError, AnomalyDetectionError.
"""

import logging

from app.application.trading.dtos import AnomalyResult, DetectAnomaliesQuery
from app.domain.trading.ports import AnomalyDetectionPort

logger = logging.getLogger(__name__)


class DetectAnomaliesUseCase:
    """Orchestrates anomaly detection for a given stock symbol.

    Delegates to the AnomalyDetectionPort and maps
    domain alerts to application DTOs.
    """

    def __init__(self, anomaly_port: AnomalyDetectionPort) -> None:
        self._anomaly_port = anomaly_port

    def execute(self, query: DetectAnomaliesQuery) -> list[AnomalyResult]:
        """Run the anomaly detection use case.

        Args:
            query: The anomaly detection request containing the symbol.

        Returns:
            A list of detected anomalies.
        """
        logger.info("Detecting anomalies for symbol=%s", query.symbol)

        # TODO: call the anomaly detection port and map results to DTOs
        alerts = self._anomaly_port.detect(symbol=query.symbol)

        return [
            AnomalyResult(
                id=alert.id,
                symbol=alert.symbol,
                anomaly_type=alert.anomaly_type,
                severity=alert.severity,
                description=alert.description,
            )
            for alert in alerts
        ]
