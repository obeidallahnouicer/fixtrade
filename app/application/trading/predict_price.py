"""
Use case: Predict stock prices for a BVMT-listed symbol.

Input: PredictPriceCommand (symbol, horizon_days)
Output: list[PredictPriceResult]
Side effects: None.
Failure cases: SymbolNotFoundError, InvalidHorizonError.
"""

import logging

from app.application.trading.dtos import PredictPriceCommand, PredictPriceResult
from app.domain.trading.errors import InvalidHorizonError
from app.domain.trading.ports import PricePredictionPort

logger = logging.getLogger(__name__)

MIN_HORIZON = 1
MAX_HORIZON = 5


class PredictPriceUseCase:
    """Orchestrates price prediction for a given stock symbol.

    Validates the prediction horizon and delegates to the
    PricePredictionPort for actual ML inference.
    """

    def __init__(self, prediction_port: PricePredictionPort) -> None:
        self._prediction_port = prediction_port

    def execute(self, command: PredictPriceCommand) -> list[PredictPriceResult]:
        """Run the price prediction use case.

        Args:
            command: The prediction request containing symbol and horizon.

        Returns:
            A list of predicted price points for each future trading day.

        Raises:
            InvalidHorizonError: If horizon is outside 1-5 range.
        """
        if not (MIN_HORIZON <= command.horizon_days <= MAX_HORIZON):
            raise InvalidHorizonError(command.horizon_days)

        logger.info(
            "Predicting price for symbol=%s, horizon=%d",
            command.symbol,
            command.horizon_days,
        )

        # TODO: call the prediction port and map results to DTOs
        predictions = self._prediction_port.predict(
            symbol=command.symbol,
            horizon_days=command.horizon_days,
        )

        return [
            PredictPriceResult(
                symbol=p.symbol,
                target_date=p.target_date,
                predicted_close=p.predicted_close,
                confidence_lower=p.confidence_lower,
                confidence_upper=p.confidence_upper,
            )
            for p in predictions
        ]
