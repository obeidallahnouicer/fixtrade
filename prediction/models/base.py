"""
Abstract base class for all prediction models.

Defines the Strategy Pattern interface that all concrete models
(LSTM, XGBoost, Prophet) must implement.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""

    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0
    directional_accuracy: float = 0.0
    r_squared: float = 0.0

    def to_dict(self) -> dict:
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape,
            "directional_accuracy": self.directional_accuracy,
            "r_squared": self.r_squared,
        }

    def __str__(self) -> str:
        return (
            f"MAE={self.mae:.4f} | RMSE={self.rmse:.4f} | "
            f"MAPE={self.mape:.2%} | DirAcc={self.directional_accuracy:.2%} | "
            f"R²={self.r_squared:.4f}"
        )


class BasePredictionModel(ABC):
    """Abstract interface for all prediction models.

    Concrete implementations: LSTMPredictor, XGBoostPredictor, ProphetPredictor.

    Methods:
        fit:            Train the model.
        predict:        Generate point predictions.
        predict_proba:  Generate predictions with confidence intervals.
        get_metrics:    Return evaluation metrics from the last evaluation.
        save_model:     Persist model artifacts to disk.
        load_model:     Load model artifacts from disk.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._metrics = ModelMetrics()
        self._is_fitted = False

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "BasePredictionModel":
        """Train the model.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Optional validation features (for early stopping).
            y_val: Optional validation targets.

        Returns:
            self for method chaining.
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point predictions.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of predicted values.
        """
        ...

    def predict_proba(
        self, X: pd.DataFrame, confidence_level: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions with confidence intervals.

        Default implementation: point prediction ± 2 * std of training residuals.

        Args:
            X: Feature DataFrame.
            confidence_level: Confidence level for intervals (default 0.95).

        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds).
        """
        preds = self.predict(X)
        # Default: use RMSE as uncertainty proxy
        margin = 2.0 * self._metrics.rmse if self._metrics.rmse > 0 else 0.0
        lower = preds - margin
        upper = preds + margin
        return preds, lower, upper

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> ModelMetrics:
        """Evaluate the model on a test set.

        Updates internal metrics and returns them.
        """
        preds = self.predict(X_test)
        self._metrics = self._compute_metrics(y_test.values, preds)
        logger.info("[%s] Evaluation: %s", self.name, self._metrics)
        return self._metrics

    def get_metrics(self) -> ModelMetrics:
        """Return the latest evaluation metrics."""
        return self._metrics

    @abstractmethod
    def save_model(self, path: Path) -> None:
        """Persist model artifacts to disk."""
        ...

    @abstractmethod
    def load_model(self, path: Path) -> None:
        """Load model artifacts from disk."""
        ...

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    # ------------------------------------------------------------------
    # Shared metric computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """Compute standard regression + directional accuracy metrics."""
        # Drop NaN pairs (e.g. LSTM pads first seq_len rows with NaN)
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[valid]
        y_pred = y_pred[valid]

        if len(y_true) == 0:
            return ModelMetrics()

        residuals = y_true - y_pred
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))

        # MAPE (avoid division by zero)
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs(residuals[mask] / y_true[mask]))
        else:
            mape = 0.0

        # Directional accuracy: did we predict the direction of change correctly?
        if len(y_true) > 1:
            actual_dir = np.sign(np.diff(y_true))
            pred_dir = np.sign(np.diff(y_pred))
            dir_acc = np.mean(actual_dir == pred_dir)
        else:
            dir_acc = 0.0

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return ModelMetrics(
            mae=float(mae),
            rmse=float(rmse),
            mape=float(mape),
            directional_accuracy=float(dir_acc),
            r_squared=float(r2),
        )
