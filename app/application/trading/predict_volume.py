"""
Use case: Predict daily transaction volume.

Orchestrates volume prediction by delegating to the prediction port.
"""

import logging

from app.application.trading.dtos import PredictVolumeCommand, PredictVolumeResult
from app.domain.trading.ports import PricePredictionPort

logger = logging.getLogger(__name__)


class PredictVolumeUseCase:
    """Application service for volume prediction."""

    def __init__(self, prediction_port: PricePredictionPort) -> None:
        self._prediction_port = prediction_port

    def execute(self, command: PredictVolumeCommand) -> list[PredictVolumeResult]:
        """Execute the volume prediction use case.

        Args:
            command: Input DTO with symbol and horizon.

        Returns:
            List of PredictVolumeResult DTOs.
        """
        entities = self._prediction_port.predict_volume(
            symbol=command.symbol,
            horizon_days=command.horizon_days,
        )
        return [
            PredictVolumeResult(
                symbol=e.symbol,
                target_date=e.target_date,
                predicted_volume=e.predicted_volume,
            )
            for e in entities
        ]
