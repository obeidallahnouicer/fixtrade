"""
Adapter: Anomaly detection engine.

Implements AnomalyDetectionPort.
Responsible for running anomaly detection algorithms on market data.
"""

from app.domain.trading.entities import AnomalyAlert
from app.domain.trading.ports import AnomalyDetectionPort


class AnomalyDetectionAdapter(AnomalyDetectionPort):
    """Concrete adapter for market anomaly detection.

    Implements the AnomalyDetectionPort defined in the domain layer.
    In production, this will run statistical or ML-based anomaly detection.
    """

    def __init__(self) -> None:
        # TODO: inject anomaly detection model or configuration
        pass

    def detect(self, symbol: str) -> list[AnomalyAlert]:
        """Return list of detected anomalies for a symbol.

        Args:
            symbol: BVMT stock ticker.

        Returns:
            List of AnomalyAlert entities.
        """
        # TODO: run anomaly detection on recent market data
        raise NotImplementedError("AnomalyDetectionAdapter.detect")
