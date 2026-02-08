"""
XGBoost-based liquidity probability classifier.

Predicts the probability of high / medium / low liquidity for the
next trading day.  Liquidity tiers are derived from daily volume:

    high   = volume ≥ 10,000
    medium = 1,000 ≤ volume < 10,000
    low    = volume < 1,000

Output: probability vector [P(low), P(medium), P(high)] for each row.
"""

import logging
import pickle
from dataclasses import dataclass
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

TIER_NAMES = ("low", "medium", "high")


@dataclass
class LiquidityProbability:
    """Container for a single liquidity forecast."""

    prob_low: float
    prob_medium: float
    prob_high: float

    @property
    def predicted_tier(self) -> str:
        probs = [self.prob_low, self.prob_medium, self.prob_high]
        return TIER_NAMES[int(np.argmax(probs))]

    def to_dict(self) -> dict[str, float]:
        return {
            "prob_low": round(self.prob_low, 4),
            "prob_medium": round(self.prob_medium, 4),
            "prob_high": round(self.prob_high, 4),
            "predicted_tier": self.predicted_tier,
        }


class LiquidityClassifier(BasePredictionModel):
    """Multi-class XGBoost classifier for next-day liquidity tier.

    Classes:
        0 = low    (volume < 1,000)
        1 = medium (1,000 ≤ volume < 10,000)
        2 = high   (volume ≥ 10,000)
    """

    def __init__(
        self,
        n_estimators: int = 250,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.85,
        colsample_bytree: float = 0.75,
    ) -> None:
        super().__init__(name="LiquidityXGB")
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._subsample = subsample
        self._colsample_bytree = colsample_bytree
        self._model: "xgb.XGBClassifier | None" = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "LiquidityClassifier":
        """Train the multi-class liquidity classifier."""
        if not HAS_XGB:
            raise RuntimeError("XGBoost is required for LiquidityClassifier.")

        self._feature_names = list(X_train.columns)

        # Ensure integer labels
        y_train_int = np.asarray(y_train, dtype=int)

        self._model = xgb.XGBClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            subsample=self._subsample,
            colsample_bytree=self._colsample_bytree,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            verbosity=0,
        )

        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_val.values, np.asarray(y_val, dtype=int))]

        self._model.fit(
            X_train.values,
            y_train_int,
            eval_set=eval_set if eval_set else None,
            verbose=False,
        )

        self._is_fitted = True

        # Log class distribution
        from collections import Counter
        dist = Counter(y_train_int)
        logger.info(
            "[LiquidityXGB] Training complete. Class distribution: %s",
            {TIER_NAMES[k]: v for k, v in sorted(dist.items())},
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels (0, 1, 2)."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("LiquidityClassifier is not fitted.")
        return self._model.predict(X.values)

    def predict_proba_tiers(self, X: pd.DataFrame) -> list[LiquidityProbability]:
        """Return per-row probability vectors for all three tiers.

        Returns:
            List of LiquidityProbability, one per row in X.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("LiquidityClassifier is not fitted.")
        proba = self._model.predict_proba(X.values)  # shape (n, 3)
        results = []
        for row in proba:
            # Ensure we have exactly 3 classes; pad if needed
            p = list(row) + [0.0] * (3 - len(row))
            results.append(LiquidityProbability(
                prob_low=float(p[0]),
                prob_medium=float(p[1]),
                prob_high=float(p[2]),
            ))
        return results

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> ModelMetrics:
        """Evaluate with classification accuracy stored in directional_accuracy."""
        preds = self.predict(X_test)
        y_true = np.asarray(y_test, dtype=int)

        # Filter NaN
        valid = ~(np.isnan(y_true) | np.isnan(preds))
        y_true = y_true[valid]
        preds = preds[valid]

        accuracy = float(np.mean(y_true == preds)) if len(y_true) > 0 else 0.0

        self._metrics = ModelMetrics(
            mae=0.0,
            rmse=0.0,
            mape=0.0,
            directional_accuracy=accuracy,
            r_squared=0.0,
        )
        logger.info(
            "[LiquidityXGB] Accuracy=%.2f%% (%d samples)",
            accuracy * 100,
            len(y_true),
        )
        return self._metrics

    def save_model(self, path: Path) -> None:
        """Save classifier to disk."""
        path.mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            self._model.save_model(str(path / "liquidity_xgb_model.json"))
        meta = {"feature_names": self._feature_names}
        with open(path / "liquidity_xgb_meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        logger.info("[LiquidityXGB] Model saved to %s", path)

    def load_model(self, path: Path) -> None:
        """Load classifier from disk."""
        if not HAS_XGB:
            raise RuntimeError("XGBoost is required.")
        self._model = xgb.XGBClassifier()
        self._model.load_model(str(path / "liquidity_xgb_model.json"))
        with open(path / "liquidity_xgb_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self._feature_names = meta["feature_names"]
        self._is_fitted = True
        logger.info("[LiquidityXGB] Model loaded from %s", path)
