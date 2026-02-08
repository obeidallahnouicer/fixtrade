"""
XGBoost-based daily volume prediction model.

Predicts future daily transaction volume (quantite_negociee) using
the same feature set as the price predictor. Volume is predicted
as a regression target (continuous positive value).

The model uses log-transformed volume to handle the skewed distribution
typical of market microstructure data.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from prediction.models.base import BasePredictionModel, ModelMetrics


class VolumePredictor(BasePredictionModel):
    """XGBoost regressor for daily transaction volume prediction.

    Predicts log(volume) internally and converts back to raw volume
    on output for numerical stability.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.85,
        colsample_bytree: float = 0.75,
        early_stopping_rounds: int = 20,
    ) -> None:
        super().__init__(name="VolumeXGB")
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._subsample = subsample
        self._colsample_bytree = colsample_bytree
        self._early_stopping_rounds = early_stopping_rounds
        self._model: "xgb.XGBRegressor | None" = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "VolumePredictor":
        """Train on log-transformed volume."""
        if not HAS_XGB:
            raise RuntimeError("XGBoost is required for VolumePredictor.")

        self._feature_names = list(X_train.columns)

        # Log-transform targets (add 1 to avoid log(0))
        y_arr = np.asarray(y_train, dtype=float)
        y_train_log = np.log1p(np.clip(y_arr, 0, None))
        if y_val is not None:
            y_val_arr = np.asarray(y_val, dtype=float)
            y_val_log = np.log1p(np.clip(y_val_arr, 0, None))
        else:
            y_val_log = None

        self._model = xgb.XGBRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            subsample=self._subsample,
            colsample_bytree=self._colsample_bytree,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            verbosity=0,
        )

        eval_set = []
        if X_val is not None and y_val_log is not None:
            eval_set = [(X_val.values, y_val_log)]

        self._model.fit(
            X_train.values,
            y_train_log,
            eval_set=eval_set if eval_set else None,
            verbose=False,
        )

        self._is_fitted = True
        logger.info("[VolumeXGB] Training complete.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict raw volume (inverse log-transform)."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("VolumePredictor is not fitted.")
        log_preds = self._model.predict(X.values)
        return np.expm1(log_preds).clip(min=0)

    def save_model(self, path: Path) -> None:
        """Save volume model to disk."""
        path.mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            self._model.save_model(str(path / "volume_xgb_model.json"))
        meta = {"feature_names": self._feature_names}
        with open(path / "volume_xgb_meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        logger.info("[VolumeXGB] Model saved to %s", path)

    def load_model(self, path: Path) -> None:
        """Load volume model from disk."""
        if not HAS_XGB:
            raise RuntimeError("XGBoost is required.")
        self._model = xgb.XGBRegressor()
        self._model.load_model(str(path / "volume_xgb_model.json"))
        with open(path / "volume_xgb_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self._feature_names = meta["feature_names"]
        self._is_fitted = True
        logger.info("[VolumeXGB] Model loaded from %s", path)
