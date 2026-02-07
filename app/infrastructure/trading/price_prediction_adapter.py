"""
Adapter: Price prediction ML model.

Implements PricePredictionPort.
Responsible for loading and running the ML model for price forecasting.
"""

from app.domain.trading.entities import PricePrediction
from app.domain.trading.ports import PricePredictionPort


class PricePredictionAdapter(PricePredictionPort):
    """Concrete adapter for ML-based price prediction.

    Implements the PricePredictionPort defined in the domain layer.
    In production, this will load a trained model and run inference.
    """

    def __init__(self) -> None:
        # TODO: load trained ML model (e.g., from file or model registry)
        pass

    def predict(
        self, symbol: str, horizon_days: int
    ) -> list[PricePrediction]:
        """Return predicted closing prices for the next N trading days.

        Args:
            symbol: BVMT stock ticker.
            horizon_days: Number of future days to predict (1-5).

        Returns:
            List of PricePrediction entities ordered by target date.
        """
        # TODO: run model inference and return predictions
        raise NotImplementedError("PricePredictionAdapter.predict")
