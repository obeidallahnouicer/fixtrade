"""
Use case: Predict liquidity tier probabilities.

Orchestrates liquidity probability forecasting by delegating to the prediction port.
"""

import logging
from decimal import Decimal

from app.application.trading.dtos import (
    PredictLiquidityCommand,
    PredictLiquidityResult,
)
from app.domain.trading.ports import PricePredictionPort

logger = logging.getLogger(__name__)


class PredictLiquidityUseCase:
    """Application service for liquidity probability prediction."""

    def __init__(self, prediction_port: PricePredictionPort) -> None:
        self._prediction_port = prediction_port

    def execute(
        self, command: PredictLiquidityCommand
    ) -> list[PredictLiquidityResult]:
        """Execute the liquidity probability use case.

        Args:
            command: Input DTO with symbol and horizon.

        Returns:
            List of PredictLiquidityResult DTOs with probability vectors.
        """
        entities = self._prediction_port.predict_liquidity(
            symbol=command.symbol,
            horizon_days=command.horizon_days,
        )
        return [
            PredictLiquidityResult(
                symbol=e.symbol,
                target_date=e.target_date,
                prob_low=Decimal(str(e.prob_low)),
                prob_medium=Decimal(str(e.prob_medium)),
                prob_high=Decimal(str(e.prob_high)),
                predicted_tier=e.predicted_tier,
            )
            for e in entities
        ]
