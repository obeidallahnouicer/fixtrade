"""
Prediction inference service.

Provides the main prediction interface used by the FastAPI adapter.
Implements the full inference flow:
1. Check Redis cache
2. Fetch features from Feature Store
3. Load models from Model Registry
4. Run inference (parallel execution via ensemble)
5. Compute confidence intervals
6. Cache result
7. Return prediction

SLA: Cache HIT < 50ms, Cache MISS < 2000ms
"""

import logging
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from prediction.config import PredictionConfig, config
from prediction.models.ensemble import EnsemblePredictor, EnsemblePrediction
from prediction.models.lstm import LSTMPredictor
from prediction.models.prophet_model import ProphetPredictor
from prediction.models.xgboost_model import XGBoostPredictor
from prediction.utils.cache import CacheClient

logger = logging.getLogger(__name__)


class PredictionResult:
    """A single prediction result for one future day."""

    def __init__(
        self,
        symbol: str,
        target_date: date,
        predicted_close: float,
        confidence_lower: float,
        confidence_upper: float,
        model_name: str = "ensemble",
        confidence_score: float = 0.0,
    ) -> None:
        self.symbol = symbol
        self.target_date = target_date
        self.predicted_close = Decimal(str(round(predicted_close, 3)))
        self.confidence_lower = Decimal(str(round(confidence_lower, 3)))
        self.confidence_upper = Decimal(str(round(confidence_upper, 3)))
        self.model_name = model_name
        self.confidence_score = confidence_score


class PredictionService:
    """Main inference service for stock price predictions.

    Orchestrates the full prediction flow with caching,
    model loading, and confidence interval computation.
    """

    def __init__(
        self,
        cfg: PredictionConfig | None = None,
        cache: CacheClient | None = None,
    ) -> None:
        self._cfg = cfg or config
        self._cache = cache or CacheClient(redis_url=self._cfg.redis.url)
        self._ensemble: EnsemblePredictor | None = None
        self._models_dir = self._cfg.paths.models_dir

    def predict(
        self,
        symbol: str,
        horizon_days: int = 1,
        model: str = "ensemble",
        confidence_level: float = 0.95,
    ) -> list[PredictionResult]:
        """Generate price predictions for a symbol.

        Args:
            symbol: BVMT ticker symbol.
            horizon_days: Number of future trading days (1-5).
            model: Model to use ('ensemble', 'lstm', 'xgboost', 'prophet').
            confidence_level: Confidence interval level (default 0.95).

        Returns:
            List of PredictionResult, one per future trading day.
        """
        # 1. Check cache
        cache_key_data = self._cache.get_prediction(symbol, model)
        if cache_key_data and self._is_cache_valid(cache_key_data, horizon_days):
            logger.info("[Cache HIT] %s/%s", symbol, model)
            return self._deserialize_predictions(cache_key_data)

        logger.info("[Cache MISS] %s/%s — running inference", symbol, model)

        # 2. Load the ensemble (or specific model)
        ensemble = self._get_ensemble()

        # 3. Fetch latest features
        features = self._get_latest_features(symbol)
        if features is None or features.empty:
            logger.warning("No features available for %s. Using fallback.", symbol)
            return self._fallback_prediction(symbol, horizon_days)

        # 4. Run inference
        results: list[PredictionResult] = []
        base_date = date.today()

        for day_offset in range(1, horizon_days + 1):
            target_date = self._next_trading_day(base_date, day_offset)

            try:
                if model == "ensemble" and ensemble is not None:
                    ens_pred = ensemble.predict_single(features)
                    results.append(PredictionResult(
                        symbol=symbol,
                        target_date=target_date,
                        predicted_close=ens_pred.predicted_value,
                        confidence_lower=ens_pred.confidence_lower,
                        confidence_upper=ens_pred.confidence_upper,
                        model_name="ensemble",
                        confidence_score=ens_pred.confidence_score,
                    ))
                else:
                    preds, lower, upper = ensemble.predict_proba(
                        features, confidence_level
                    )
                    idx = min(day_offset - 1, len(preds) - 1)
                    results.append(PredictionResult(
                        symbol=symbol,
                        target_date=target_date,
                        predicted_close=float(preds[idx]),
                        confidence_lower=float(lower[idx]),
                        confidence_upper=float(upper[idx]),
                        model_name=model,
                    ))
            except Exception:
                logger.exception("Inference failed for %s day %d", symbol, day_offset)
                results.append(self._fallback_prediction(symbol, 1)[0])

        # 5. Cache result
        self._cache.set_prediction(
            symbol, self._serialize_predictions(results), model
        )

        return results

    def _get_ensemble(self) -> EnsemblePredictor:
        """Load or return the cached ensemble model."""
        if self._ensemble is not None and self._ensemble.is_fitted:
            return self._ensemble

        self._ensemble = EnsemblePredictor(
            models={
                "LSTM": LSTMPredictor(),
                "XGBoost": XGBoostPredictor(),
                "Prophet": ProphetPredictor(),
            }
        )

        ensemble_path = self._models_dir / "ensemble"
        if ensemble_path.exists():
            try:
                self._ensemble.load_model(ensemble_path)
                logger.info("Ensemble model loaded from registry.")
            except Exception:
                logger.warning("Failed to load ensemble. Models not trained.")
        else:
            logger.warning(
                "No trained models found at %s. Predictions will use fallback.",
                ensemble_path,
            )

        return self._ensemble

    def _get_latest_features(self, symbol: str) -> pd.DataFrame | None:
        """Fetch latest features from cache or compute on-the-fly."""
        # Try Redis feature store first
        today_str = date.today().isoformat()
        cached = self._cache.get_features(symbol, today_str)
        if cached:
            return pd.DataFrame([cached])

        # Fallback: load from Silver layer Parquet
        silver_path = self._cfg.paths.base_dir / "silver"
        if silver_path.exists():
            try:
                parquet_files = sorted(silver_path.rglob("*.parquet"))
                for pf in parquet_files:
                    df = pd.read_parquet(pf)
                    if "code" in df.columns:
                        ticker_df = df[df["code"] == symbol]
                        if not ticker_df.empty:
                            return ticker_df.sort_values("seance").tail(1)
            except Exception:
                logger.warning("Failed to load features for %s", symbol)

        return None

    @staticmethod
    def _fallback_prediction(
        symbol: str, horizon_days: int
    ) -> list[PredictionResult]:
        """Return a placeholder prediction when models are unavailable."""
        results = []
        base = date.today()
        for i in range(1, horizon_days + 1):
            results.append(PredictionResult(
                symbol=symbol,
                target_date=base + timedelta(days=i),
                predicted_close=0.0,
                confidence_lower=0.0,
                confidence_upper=0.0,
                model_name="fallback",
                confidence_score=0.0,
            ))
        return results

    @staticmethod
    def _next_trading_day(from_date: date, days_ahead: int) -> date:
        """Compute the next N-th trading day (skip weekends)."""
        current = from_date
        count = 0
        while count < days_ahead:
            current += timedelta(days=1)
            if current.weekday() < 5:  # Mon-Fri
                count += 1
        return current

    @staticmethod
    def _is_cache_valid(data: dict, horizon_days: int) -> bool:
        """Check if cached predictions cover the requested horizon."""
        predictions = data.get("predictions", [])
        return len(predictions) >= horizon_days

    @staticmethod
    def _serialize_predictions(results: list[PredictionResult]) -> dict:
        """Serialize predictions for cache storage."""
        return {
            "predictions": [
                {
                    "symbol": r.symbol,
                    "target_date": r.target_date.isoformat(),
                    "predicted_close": str(r.predicted_close),
                    "confidence_lower": str(r.confidence_lower),
                    "confidence_upper": str(r.confidence_upper),
                    "model_name": r.model_name,
                    "confidence_score": r.confidence_score,
                }
                for r in results
            ]
        }

    @staticmethod
    def _deserialize_predictions(data: dict) -> list[PredictionResult]:
        """Deserialize predictions from cache."""
        return [
            PredictionResult(
                symbol=p["symbol"],
                target_date=date.fromisoformat(p["target_date"]),
                predicted_close=float(p["predicted_close"]),
                confidence_lower=float(p["confidence_lower"]),
                confidence_upper=float(p["confidence_upper"]),
                model_name=p.get("model_name", "cached"),
                confidence_score=float(p.get("confidence_score", 0)),
            )
            for p in data.get("predictions", [])
        ]
