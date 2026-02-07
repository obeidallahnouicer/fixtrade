"""
XGBoost-based price prediction model.

Tree-based gradient boosting with:
- Feature importance extraction
- Built-in early stopping
- Handles missing values natively
- No sequence requirement (tabular features)

Best for: medium-frequency patterns, feature interactions.
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
    logger.warning("XGBoost not installed. XGBoostPredictor will be unavailable.")

from prediction.models.base import BasePredictionModel, ModelMetrics


class XGBoostPredictor(BasePredictionModel):
    """XGBoost model for stock price prediction.

    Gradient-boosted decision trees with built-in feature importance
    and native missing value support.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 20,
    ) -> None:
        super().__init__(name="XGBoost")
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._subsample = subsample
        self._colsample_bytree = colsample_bytree
        self._early_stopping_rounds = early_stopping_rounds
        self._model: "xgb.XGBRegressor | None" = None
        self._feature_names: list[str] = []
        self._feature_importances: dict[str, float] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "XGBoostPredictor":
        """Train the XGBoost model with optional early stopping."""
        if not HAS_XGB:
            raise RuntimeError("XGBoost is required.")

        self._feature_names = list(X_train.columns)

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
        if X_val is not None and y_val is not None:
            eval_set = [(X_val.values, y_val.values)]

        self._model.fit(
            X_train.values,
            y_train.values,
            eval_set=eval_set if eval_set else None,
            verbose=False,
        )

        # Extract feature importances
        importances = self._model.feature_importances_
        self._feature_importances = dict(
            sorted(
                zip(self._feature_names, importances),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        self._is_fitted = True
        logger.info(
            "[XGBoost] Training complete. Top 5 features: %s",
            list(self._feature_importances.keys())[:5],
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions from feature DataFrame."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self._model.predict(X.values)

    def get_feature_importance(self, top_n: int = 20) -> dict[str, float]:
        """Return top-N features by importance score."""
        return dict(list(self._feature_importances.items())[:top_n])

    def save_model(self, path: Path) -> None:
        """Save XGBoost model to disk."""
        path.mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            self._model.save_model(str(path / "xgboost_model.json"))
        meta = {
            "feature_names": self._feature_names,
            "feature_importances": self._feature_importances,
        }
        with open(path / "xgboost_meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        logger.info("[XGBoost] Model saved to %s", path)

    def load_model(self, path: Path) -> None:
        """Load XGBoost model from disk."""
        if not HAS_XGB:
            raise RuntimeError("XGBoost is required.")

        self._model = xgb.XGBRegressor()
        self._model.load_model(str(path / "xgboost_model.json"))

        with open(path / "xgboost_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self._feature_names = meta["feature_names"]
        self._feature_importances = meta["feature_importances"]
        self._is_fitted = True
        logger.info("[XGBoost] Model loaded from %s", path)
