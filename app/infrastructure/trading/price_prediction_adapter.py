"""
Adapter: Price prediction ML model.

Implements PricePredictionPort.
Delegates to the prediction module's inference service
which manages the ensemble ML pipeline (LSTM + XGBoost + Prophet).
"""

import logging

from app.domain.trading.entities import PricePrediction
from app.domain.trading.ports import PricePredictionPort

logger = logging.getLogger(__name__)


class PricePredictionAdapter(PricePredictionPort):
    """Concrete adapter for ML-based price prediction.

    Implements the PricePredictionPort defined in the domain layer.
    Delegates to the prediction module's PredictionService which
    manages model loading, caching, and ensemble inference.
    """

    def __init__(self) -> None:
        try:
            from prediction.inference import PredictionService
            self._service = PredictionService()
        except ImportError:
            logger.warning(
                "prediction module not available. "
                "Install ML dependencies (numpy, pandas, torch, xgboost, prophet)."
            )
            self._service = None

    def predict(
        self, symbol: str, horizon_days: int
    ) -> list[PricePrediction]:
        """Return predicted closing prices for the next N trading days.

        Delegates to the prediction module's ensemble engine.
        Falls back to empty results if the module is unavailable.

        Args:
            symbol: BVMT stock ticker.
            horizon_days: Number of future days to predict (1-5).

        Returns:
            List of PricePrediction entities ordered by target date.
        """
        if self._service is None:
            logger.error(
                "Prediction service unavailable. "
                "Ensure ML dependencies are installed."
            )
            return []

        results = self._service.predict(
            symbol=symbol,
            horizon_days=horizon_days,
            model="ensemble",
            confidence_level=0.95,
        )

        return [
            PricePrediction(
                symbol=r.symbol,
                target_date=r.target_date,
                predicted_close=r.predicted_close,
                confidence_lower=r.confidence_lower,
                confidence_upper=r.confidence_upper,
            )
            for r in results
        ]
