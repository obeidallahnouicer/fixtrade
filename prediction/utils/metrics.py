"""
Model evaluation metrics and monitoring.

Provides:
- Standard regression metrics (MAE, RMSE, MAPE, R²)
- Directional accuracy
- Prediction drift detection
- Calibration score for confidence intervals
- Rolling performance tracking
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PerformanceReport:
    """Comprehensive model performance report."""

    model_name: str
    evaluation_date: date
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float
    r_squared: float
    calibration_score: float
    mean_prediction: float
    mean_actual: float
    drift_detected: bool
    sample_size: int

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "evaluation_date": self.evaluation_date.isoformat(),
            "mae": round(self.mae, 6),
            "rmse": round(self.rmse, 6),
            "mape": round(self.mape, 4),
            "directional_accuracy": round(self.directional_accuracy, 4),
            "r_squared": round(self.r_squared, 4),
            "calibration_score": round(self.calibration_score, 4),
            "drift_detected": self.drift_detected,
            "sample_size": self.sample_size,
        }


class ModelMonitor:
    """Tracks model performance over time and detects degradation.

    Triggers alerts when:
    - RMSE > threshold × 1.5
    - Directional accuracy < 50%
    - Prediction drift detected
    """

    def __init__(
        self,
        rmse_threshold: float = 0.02,
        dir_acc_threshold: float = 0.50,
        drift_window: int = 30,
    ) -> None:
        self._rmse_threshold = rmse_threshold
        self._dir_acc_threshold = dir_acc_threshold
        self._drift_window = drift_window
        self._history: list[PerformanceReport] = []

    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_lower: np.ndarray | None = None,
        y_upper: np.ndarray | None = None,
    ) -> PerformanceReport:
        """Run full evaluation and produce a performance report.

        Args:
            model_name: Name of the model being evaluated.
            y_true: Actual values.
            y_pred: Predicted values.
            y_lower: Lower confidence bounds (optional).
            y_upper: Upper confidence bounds (optional).

        Returns:
            Comprehensive PerformanceReport.
        """
        residuals = y_true - y_pred
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        # MAPE
        mask = y_true != 0
        mape = float(np.mean(np.abs(residuals[mask] / y_true[mask]))) if mask.any() else 0.0

        # Directional accuracy
        if len(y_true) > 1:
            actual_dir = np.sign(np.diff(y_true))
            pred_dir = np.sign(np.diff(y_pred))
            dir_acc = float(np.mean(actual_dir == pred_dir))
        else:
            dir_acc = 0.0

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        # Calibration score
        calibration = 0.0
        if y_lower is not None and y_upper is not None:
            in_interval = (y_true >= y_lower) & (y_true <= y_upper)
            calibration = float(np.mean(in_interval))

        # Drift detection
        mean_pred = float(np.mean(y_pred))
        mean_actual = float(np.mean(y_true))
        drift = abs(mean_pred - mean_actual) / abs(mean_actual) > 0.05 if mean_actual != 0 else False

        report = PerformanceReport(
            model_name=model_name,
            evaluation_date=date.today(),
            mae=mae,
            rmse=rmse,
            mape=mape,
            directional_accuracy=dir_acc,
            r_squared=r2,
            calibration_score=calibration,
            mean_prediction=mean_pred,
            mean_actual=mean_actual,
            drift_detected=drift,
            sample_size=len(y_true),
        )

        self._history.append(report)
        self._check_alerts(report)

        return report

    def get_rolling_performance(
        self, model_name: str, window: int = 30
    ) -> list[PerformanceReport]:
        """Get the last N performance reports for a model."""
        model_history = [
            r for r in self._history if r.model_name == model_name
        ]
        return model_history[-window:]

    def should_retrain(self, model_name: str) -> bool:
        """Determine if a model needs retraining.

        True if:
        - RMSE exceeds 1.5× threshold
        - Directional accuracy below 50%
        - Drift detected in 3+ consecutive evaluations
        """
        recent = self.get_rolling_performance(model_name, window=5)
        if not recent:
            return False

        latest = recent[-1]

        if latest.rmse > self._rmse_threshold * 1.5:
            logger.warning(
                "[Monitor] %s RMSE (%.4f) exceeds threshold.",
                model_name,
                latest.rmse,
            )
            return True

        if latest.directional_accuracy < self._dir_acc_threshold:
            logger.warning(
                "[Monitor] %s dir_acc (%.2f%%) below threshold.",
                model_name,
                latest.directional_accuracy * 100,
            )
            return True

        drift_count = sum(1 for r in recent[-3:] if r.drift_detected)
        if drift_count >= 3:
            logger.warning(
                "[Monitor] %s shows persistent drift.", model_name
            )
            return True

        return False

    def _check_alerts(self, report: PerformanceReport) -> None:
        """Log alerts for concerning metrics."""
        if report.rmse > self._rmse_threshold * 1.5:
            logger.warning(
                "⚠️  [%s] RMSE=%.4f exceeds 1.5× threshold (%.4f).",
                report.model_name,
                report.rmse,
                self._rmse_threshold,
            )
        if report.directional_accuracy < self._dir_acc_threshold:
            logger.warning(
                "⚠️  [%s] Directional accuracy=%.1f%% below %.0f%%.",
                report.model_name,
                report.directional_accuracy * 100,
                self._dir_acc_threshold * 100,
            )
        if report.drift_detected:
            logger.warning(
                "⚠️  [%s] Prediction drift detected.", report.model_name
            )
