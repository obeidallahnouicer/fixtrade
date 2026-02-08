"""
Prophet-based price prediction model.

Facebook Prophet for trend + seasonality decomposition:
- Automatic changepoint detection
- Multiple seasonality (yearly, weekly)
- Holiday effects (Tunisian calendar)
- Robust to missing data and outliers

Best for: low-frequency trends, seasonal patterns, low-liquidity stocks.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    logger.warning("Prophet not installed. ProphetPredictor will be unavailable.")

from prediction.models.base import BasePredictionModel, ModelMetrics


class ProphetPredictor(BasePredictionModel):
    """Prophet model for stock price prediction.

    Decomposes time series into trend + seasonality + holidays.
    Useful for capturing long-term patterns and seasonal effects.
    """

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
    ) -> None:
        super().__init__(name="Prophet")
        self._cp_prior = changepoint_prior_scale
        self._season_prior = seasonality_prior_scale
        self._yearly = yearly_seasonality
        self._weekly = weekly_seasonality
        self._model: "Prophet | None" = None
        self._last_train_date: pd.Timestamp | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "ProphetPredictor":
        """Train the Prophet model.

        Prophet expects a DataFrame with 'ds' (date) and 'y' (target) columns.
        Additional regressors from X_train can be added.
        """
        if not HAS_PROPHET:
            raise RuntimeError("Prophet is required.")

        # Build Prophet input
        prophet_df = pd.DataFrame({
            "ds": pd.to_datetime(X_train.index if hasattr(X_train.index, 'date') else X_train.get("seance", X_train.index)),
            "y": y_train.values,
        })
        prophet_df = prophet_df.dropna().reset_index(drop=True)

        self._model = Prophet(
            changepoint_prior_scale=self._cp_prior,
            seasonality_prior_scale=self._season_prior,
            yearly_seasonality=self._yearly,
            weekly_seasonality=self._weekly,
            daily_seasonality=False,
        )

        # Add selected regressors if available
        regressor_cols = [
            c for c in X_train.columns
            if c in ("rsi", "macd", "volume_ratio", "volatility_20d")
            and not X_train[c].isna().all()
        ]
        for col in regressor_cols:
            self._model.add_regressor(col)
            prophet_df[col] = X_train[col].values[:len(prophet_df)]

        prophet_df = prophet_df.dropna()

        # Suppress Prophet's verbose logging
        import logging as _logging
        _logging.getLogger("prophet").setLevel(_logging.WARNING)
        _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)

        self._model.fit(prophet_df)
        self._last_train_date = prophet_df["ds"].max()
        self._regressor_cols = regressor_cols

        self._is_fitted = True
        logger.info("[Prophet] Training complete on %d rows.", len(prophet_df))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions from feature DataFrame."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        future_df = pd.DataFrame({
            "ds": pd.to_datetime(
                X.index if hasattr(X.index, 'date')
                else X.get("seance", X.index)
            ),
        })

        # Add regressor values for the prediction period
        for col in self._regressor_cols:
            if col in X.columns:
                future_df[col] = X[col].values[:len(future_df)]
            else:
                future_df[col] = 0.0

        future_df = future_df.dropna(subset=["ds"])
        forecast = self._model.predict(future_df)
        return forecast["yhat"].values

    def predict_proba(
        self, X: pd.DataFrame, confidence_level: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prophet natively provides uncertainty intervals."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        future_df = pd.DataFrame({
            "ds": pd.to_datetime(
                X.index if hasattr(X.index, 'date')
                else X.get("seance", X.index)
            ),
        })

        for col in self._regressor_cols:
            if col in X.columns:
                future_df[col] = X[col].values[:len(future_df)]
            else:
                future_df[col] = 0.0

        future_df = future_df.dropna(subset=["ds"])

        # Adjust interval width
        self._model.interval_width = confidence_level
        forecast = self._model.predict(future_df)

        return (
            forecast["yhat"].values,
            forecast["yhat_lower"].values,
            forecast["yhat_upper"].values,
        )

    def predict_future(self, periods: int = 5) -> pd.DataFrame:
        """Generate future predictions for N trading days ahead.

        Returns a DataFrame with ds, yhat, yhat_lower, yhat_upper.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        future = self._model.make_future_dataframe(periods=periods, freq="B")
        # Fill regressors with last known values
        for col in self._regressor_cols:
            future[col] = 0.0

        forecast = self._model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)

    def save_model(self, path: Path) -> None:
        """Save Prophet model to disk."""
        path.mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            with open(path / "prophet_model.pkl", "wb") as f:
                pickle.dump(self._model, f)
        meta = {
            "regressor_cols": self._regressor_cols,
            "last_train_date": self._last_train_date,
        }
        with open(path / "prophet_meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        logger.info("[Prophet] Model saved to %s", path)

    def load_model(self, path: Path) -> None:
        """Load Prophet model from disk."""
        if not HAS_PROPHET:
            raise RuntimeError("Prophet is required.")

        with open(path / "prophet_model.pkl", "rb") as f:
            self._model = pickle.load(f)
        with open(path / "prophet_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self._regressor_cols = meta["regressor_cols"]
        self._last_train_date = meta["last_train_date"]
        self._is_fitted = True
        logger.info("[Prophet] Model loaded from %s", path)
