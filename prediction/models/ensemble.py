"""
Ensemble prediction engine.

Combines predictions from multiple base models:
- Phase 1 (MVP): Simple weighted average
- Phase 2: Dynamic weighting (context-aware)
- Phase 3: Stacking with meta-learner

Supports liquidity-tiered model selection.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from prediction.models.base import BasePredictionModel, ModelMetrics

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """Container for an ensemble prediction result."""

    predicted_value: float
    confidence_lower: float
    confidence_upper: float
    model_predictions: dict[str, float]
    model_weights: dict[str, float]
    confidence_score: float  # 0-1 based on model agreement


class LiquidityTier:
    """Determines model selection strategy based on trading volume."""

    TIER1 = "high"      # Volume > 10,000 → all 3 models
    TIER2 = "medium"    # Volume 1,000-10,000 → XGBoost + Prophet
    TIER3 = "low"       # Volume < 1,000 → Prophet only

    @staticmethod
    def classify(avg_daily_volume: float) -> str:
        if avg_daily_volume >= 10_000:
            return LiquidityTier.TIER1
        elif avg_daily_volume >= 1_000:
            return LiquidityTier.TIER2
        else:
            return LiquidityTier.TIER3


class EnsemblePredictor(BasePredictionModel):
    """Ensemble model combining LSTM, XGBoost, and Prophet.

    Implements three weighting strategies:
    1. Static weighted average (default)
    2. Dynamic weighting based on market conditions
    3. Stacking meta-learner (when trained)
    """

    def __init__(
        self,
        models: dict[str, BasePredictionModel] | None = None,
        weights: dict[str, float] | None = None,
        use_dynamic_weights: bool = False,
    ) -> None:
        super().__init__(name="Ensemble")
        self._models = models or {}
        self._weights = weights or {
            "LSTM": 0.45,
            "XGBoost": 0.35,
            "Prophet": 0.20,
        }
        self._use_dynamic = use_dynamic_weights
        self._meta_learner = None  # Phase 3: Ridge regression meta-learner

    def add_model(self, model: BasePredictionModel, weight: float = 0.0) -> None:
        """Register a base model in the ensemble."""
        self._models[model.name] = model
        if weight > 0:
            self._weights[model.name] = weight
        self._normalize_weights()

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "EnsemblePredictor":
        """Train all base models.

        Each model is trained independently. Weights are then
        optimized based on validation performance.
        """
        for name, model in self._models.items():
            logger.info("[Ensemble] Training base model: %s", name)
            try:
                model.fit(X_train, y_train, X_val, y_val)
            except Exception:
                logger.exception("[Ensemble] Failed to train %s", name)

        # Optimize weights on validation set
        if X_val is not None and y_val is not None:
            self._optimize_weights(X_val, y_val)

        self._is_fitted = True
        logger.info(
            "[Ensemble] All models trained. Weights: %s", self._weights
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions (weighted average)."""
        individual_preds = self._get_individual_predictions(X)
        if not individual_preds:
            raise RuntimeError("No base models produced predictions.")

        # Weighted average
        result = np.zeros(len(X))
        total_weight = 0.0

        for name, preds in individual_preds.items():
            w = self._weights.get(name, 0.0)
            # Handle NaN predictions
            valid_mask = ~np.isnan(preds)
            result[valid_mask] += w * preds[valid_mask]
            total_weight += w

        if total_weight > 0:
            result /= total_weight

        return result

    def predict_proba(
        self, X: pd.DataFrame, confidence_level: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate ensemble predictions with confidence intervals.

        CI = [min(predictions) - 2σ, max(predictions) + 2σ]
        """
        individual_preds = self._get_individual_predictions(X)
        if not individual_preds:
            raise RuntimeError("No base models produced predictions.")

        preds_matrix = np.array(list(individual_preds.values()))

        # Weighted mean
        weights = np.array([
            self._weights.get(name, 0.0) for name in individual_preds.keys()
        ])
        weights = weights / weights.sum()
        mean_pred = np.average(preds_matrix, axis=0, weights=weights)

        # Uncertainty from model disagreement
        std_pred = np.std(preds_matrix, axis=0)
        z_score = 1.96 if confidence_level >= 0.95 else 1.645

        lower = np.nanmin(preds_matrix, axis=0) - z_score * std_pred
        upper = np.nanmax(preds_matrix, axis=0) + z_score * std_pred

        return mean_pred, lower, upper

    def predict_single(
        self,
        X: pd.DataFrame,
        avg_daily_volume: float | None = None,
        market_volatility: float | None = None,
    ) -> EnsemblePrediction:
        """Generate a single ensemble prediction with full metadata.

        Supports dynamic weighting based on market conditions.
        """
        # Determine liquidity tier for model selection
        tier = LiquidityTier.TIER1
        if avg_daily_volume is not None:
            tier = LiquidityTier.classify(avg_daily_volume)

        # Select models based on tier
        active_models = self._select_models_for_tier(tier)

        # Dynamic weights if enabled
        if self._use_dynamic and market_volatility is not None:
            weights = self._compute_dynamic_weights(
                market_volatility, avg_daily_volume or 0
            )
        else:
            weights = {k: self._weights.get(k, 0) for k in active_models}

        # Normalize weights for active models
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # Collect predictions
        model_preds: dict[str, float] = {}
        for name in active_models:
            if name in self._models and self._models[name].is_fitted:
                try:
                    preds = self._models[name].predict(X)
                    if len(preds) > 0:
                        model_preds[name] = float(np.nanmean(preds[-1:]))
                except Exception:
                    logger.warning("[Ensemble] %s prediction failed.", name)

        if not model_preds:
            raise RuntimeError("No models produced valid predictions.")

        # Weighted average
        final_pred = sum(
            weights.get(name, 0) * val
            for name, val in model_preds.items()
        )

        # Confidence from model agreement
        pred_values = list(model_preds.values())
        std = np.std(pred_values) if len(pred_values) > 1 else 0.0
        mean_price = abs(final_pred) if final_pred != 0 else 1.0
        agreement = max(0, 1 - (std / mean_price))

        # Confidence score based on tier + agreement
        tier_base = {"high": 0.7, "medium": 0.5, "low": 0.3}
        confidence = min(1.0, tier_base.get(tier, 0.5) * (0.5 + 0.5 * agreement))

        return EnsemblePrediction(
            predicted_value=final_pred,
            confidence_lower=min(pred_values) - 2 * std,
            confidence_upper=max(pred_values) + 2 * std,
            model_predictions=model_preds,
            model_weights=weights,
            confidence_score=confidence,
        )

    def save_model(self, path: Path) -> None:
        """Save all base models and ensemble weights."""
        path.mkdir(parents=True, exist_ok=True)
        for name, model in self._models.items():
            model_path = path / name.lower()
            model.save_model(model_path)

        import json
        with open(path / "ensemble_weights.json", "w") as f:
            json.dump(self._weights, f, indent=2)
        logger.info("[Ensemble] All models saved to %s", path)

    def load_model(self, path: Path) -> None:
        """Load all base models and ensemble weights."""
        import json
        with open(path / "ensemble_weights.json", "r") as f:
            self._weights = json.load(f)

        for name, model in self._models.items():
            model_path = path / name.lower()
            if model_path.exists():
                model.load_model(model_path)
                logger.info("[Ensemble] Loaded %s", name)

        self._is_fitted = True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_individual_predictions(
        self, X: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """Collect predictions from all fitted base models."""
        predictions: dict[str, np.ndarray] = {}
        for name, model in self._models.items():
            if model.is_fitted:
                try:
                    preds = model.predict(X)
                    predictions[name] = preds
                except Exception:
                    logger.warning(
                        "[Ensemble] %s failed to predict.", name
                    )
        return predictions

    def _optimize_weights(
        self, X_val: pd.DataFrame, y_val: pd.Series
    ) -> None:
        """Optimize ensemble weights based on validation RMSE.

        Weight = 1 / RMSE_validation, then normalized.
        """
        rmse_scores: dict[str, float] = {}
        for name, model in self._models.items():
            if model.is_fitted:
                try:
                    metrics = model.evaluate(X_val, y_val)
                    if metrics.rmse > 0:
                        rmse_scores[name] = metrics.rmse
                except Exception:
                    logger.warning(
                        "[Ensemble] Could not evaluate %s on validation.", name
                    )

        if rmse_scores:
            inverse_rmse = {k: 1.0 / v for k, v in rmse_scores.items()}
            total = sum(inverse_rmse.values())
            self._weights = {k: v / total for k, v in inverse_rmse.items()}
            logger.info(
                "[Ensemble] Optimized weights: %s", self._weights
            )

    def _normalize_weights(self) -> None:
        """Ensure weights sum to 1.0."""
        total = sum(self._weights.values())
        if total > 0:
            self._weights = {k: v / total for k, v in self._weights.items()}

    @staticmethod
    def _select_models_for_tier(tier: str) -> list[str]:
        """Select which models to use based on liquidity tier."""
        if tier == LiquidityTier.TIER1:
            return ["LSTM", "XGBoost", "Prophet"]
        elif tier == LiquidityTier.TIER2:
            return ["XGBoost", "Prophet"]
        else:
            return ["Prophet"]

    @staticmethod
    def _compute_dynamic_weights(
        market_volatility: float,
        avg_daily_volume: float,
    ) -> dict[str, float]:
        """Phase 2: Context-aware dynamic weight computation.

        Adjusts weights based on current market regime.
        """
        if market_volatility > 0.03:
            # High volatility: Prophet handles well, LSTM struggles
            return {"LSTM": 0.2, "XGBoost": 0.3, "Prophet": 0.5}
        elif avg_daily_volume < 5000:
            # Low liquidity: favor trend-based
            return {"LSTM": 0.25, "XGBoost": 0.15, "Prophet": 0.6}
        elif market_volatility < 0.01:
            # Strong trend: LSTM excels
            return {"LSTM": 0.6, "XGBoost": 0.3, "Prophet": 0.1}
        else:
            # Default
            return {"LSTM": 0.45, "XGBoost": 0.35, "Prophet": 0.20}
