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

Also supports:
- Volume forecasting (next N trading days)
- Liquidity probability classification (P(high/medium/low))

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
from prediction.models.liquidity_classifier import LiquidityClassifier, LiquidityProbability
from prediction.models.lstm import LSTMPredictor
from prediction.models.prophet_model import ProphetPredictor
from prediction.models.volume_predictor import VolumePredictor
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


class VolumeResult:
    """Predicted daily transaction volume for one future day."""

    def __init__(
        self,
        symbol: str,
        target_date: date,
        predicted_volume: float,
    ) -> None:
        self.symbol = symbol
        self.target_date = target_date
        self.predicted_volume = round(predicted_volume, 0)


class LiquidityResult:
    """Liquidity probability forecast for the next trading day."""

    def __init__(
        self,
        symbol: str,
        target_date: date,
        prob_low: float,
        prob_medium: float,
        prob_high: float,
    ) -> None:
        self.symbol = symbol
        self.target_date = target_date
        self.prob_low = round(prob_low, 4)
        self.prob_medium = round(prob_medium, 4)
        self.prob_high = round(prob_high, 4)

    @property
    def predicted_tier(self) -> str:
        probs = [self.prob_low, self.prob_medium, self.prob_high]
        return ("low", "medium", "high")[int(np.argmax(probs))]


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

        # 2. Load the ensemble (or specific model) — prefer per-symbol models
        ensemble = self._get_ensemble(symbol)

        # 3. Fetch latest features
        features = self._get_latest_features(symbol)
        if features is None or features.empty:
            logger.warning("No features available for %s. Using fallback.", symbol)
            return self._fallback_prediction(symbol, horizon_days)

        # Strip metadata columns — keep only numeric feature columns
        from prediction.etl.transform.silver_to_gold import SilverToGoldTransformer
        feature_cols = SilverToGoldTransformer.get_feature_columns(features)
        features = features[feature_cols].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)

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
                results.append(PredictionResult(
                    symbol=symbol,
                    target_date=target_date,
                    predicted_close=0.0,
                    confidence_lower=0.0,
                    confidence_upper=0.0,
                    model_name="fallback",
                    confidence_score=0.0,
                ))

        # 5. Cache result
        self._cache.set_prediction(
            symbol, self._serialize_predictions(results), model
        )

        return results

    def _get_ensemble(self, symbol: str | None = None) -> EnsemblePredictor:
        """Load or return the cached ensemble model.

        Tries per-symbol models first (models/ensemble/{SYMBOL}/),
        then falls back to global models (models/ensemble/).
        """
        if self._ensemble is not None and self._ensemble.is_fitted:
            return self._ensemble

        self._ensemble = EnsemblePredictor(
            models={
                "LSTM": LSTMPredictor(),
                "XGBoost": XGBoostPredictor(),
                "Prophet": ProphetPredictor(),
            }
        )

        # Try per-symbol models first, then fall back to global
        candidates: list[Path] = []
        if symbol:
            candidates.append(self._models_dir / "ensemble" / symbol.upper())
        candidates.append(self._models_dir / "ensemble")

        loaded = False
        for ensemble_path in candidates:
            if ensemble_path.exists() and (ensemble_path / "ensemble_weights.json").exists():
                try:
                    self._ensemble.load_model(ensemble_path)
                    logger.info("Ensemble model loaded from %s", ensemble_path)
                    loaded = True
                    break
                except Exception:
                    logger.warning("Failed to load ensemble from %s.", ensemble_path)

        if not loaded:
            logger.warning(
                "No trained models found. Predictions will use fallback. "
                "Train first: python -m prediction train --symbol %s --final",
                symbol or "<SYMBOL>",
            )

        return self._ensemble

    def _get_latest_features(self, symbol: str) -> pd.DataFrame | None:
        """Fetch latest features from cache or compute on-the-fly.

        Returns enough rows for LSTM sequence creation (60+).
        Scans all Silver partitions to collect data across code variants.
        """
        # Try Redis feature store first
        today_str = date.today().isoformat()
        cached = self._cache.get_features(symbol, today_str)
        if cached:
            return pd.DataFrame([cached])

        # Fallback: load from Silver layer Parquet
        silver_path = self._cfg.paths.base_dir / "silver"
        if not silver_path.exists():
            return None

        try:
            import pyarrow.parquet as pq

            matched_frames: list[pd.DataFrame] = []
            for pf in sorted(silver_path.rglob("*.parquet")):
                # Fast check: read only the libelle column to see if this
                # partition contains data for the requested symbol
                schema = pq.read_schema(pf)
                if "libelle" in schema.names:
                    tbl = pq.read_table(pf, columns=["libelle"])
                    labels = tbl.column("libelle").to_pylist()
                    if not any(str(lb).upper() == symbol.upper() for lb in labels):
                        continue
                # Full read for matching partition
                df = pd.read_parquet(pf)

                # Reconstruct partition column from directory path
                if "code" not in df.columns:
                    for part in pf.relative_to(silver_path).parent.parts:
                        if part.startswith("code="):
                            df["code"] = part.split("=", 1)[1]
                            break

                # Match by symbol name (libelle) or by numeric code
                if "libelle" in df.columns:
                    ticker_df = df[df["libelle"].str.upper() == symbol.upper()]
                elif "code" in df.columns:
                    ticker_df = df[df["code"] == symbol]
                else:
                    continue

                if not ticker_df.empty:
                    matched_frames.append(ticker_df)

            if not matched_frames:
                return None

            combined = pd.concat(matched_frames, ignore_index=True)
            combined = combined.sort_values("seance").reset_index(drop=True)
            # Return enough rows for LSTM sequences (need 60+)
            n_rows = max(100, self._cfg.model.lstm_sequence_length + 10)
            return combined.tail(n_rows)
        except Exception:
            logger.warning("Failed to load features for %s", symbol)
            return None

    # ------------------------------------------------------------------
    # Volume Prediction
    # ------------------------------------------------------------------

    def predict_volume(
        self,
        symbol: str,
        horizon_days: int = 5,
    ) -> list[VolumeResult]:
        """Predict daily transaction volume for the next N trading days.

        Args:
            symbol: BVMT ticker symbol.
            horizon_days: Number of future trading days (1-5).

        Returns:
            List of VolumeResult, one per future trading day.
        """
        features = self._get_latest_features(symbol)
        if features is None or features.empty:
            logger.warning("No features for volume prediction on %s.", symbol)
            return []

        from prediction.etl.transform.silver_to_gold import SilverToGoldTransformer
        feature_cols = SilverToGoldTransformer.get_feature_columns(features)
        X = features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        vol_model = self._load_volume_model(symbol)
        if vol_model is None:
            logger.warning("No volume model found for %s.", symbol)
            return []

        results: list[VolumeResult] = []
        base_date = date.today()

        try:
            preds = vol_model.predict(X)
            # Use the last prediction for each horizon day
            last_pred = float(preds[-1]) if len(preds) > 0 else 0.0

            for day_offset in range(1, horizon_days + 1):
                target_date = self._next_trading_day(base_date, day_offset)
                results.append(VolumeResult(
                    symbol=symbol,
                    target_date=target_date,
                    predicted_volume=last_pred,
                ))
        except Exception:
            logger.exception("Volume prediction failed for %s", symbol)

        return results

    def _load_volume_model(self, symbol: str) -> VolumePredictor | None:
        """Load the volume prediction model from registry."""
        candidates: list[Path] = []
        if symbol:
            candidates.append(self._models_dir / "volume" / symbol.upper())
        candidates.append(self._models_dir / "volume")

        for path in candidates:
            if path.exists() and (path / "volume_xgb_model.json").exists():
                try:
                    model = VolumePredictor()
                    model.load_model(path)
                    return model
                except Exception:
                    logger.warning("Failed to load volume model from %s", path)
        return None

    # ------------------------------------------------------------------
    # Liquidity Probability
    # ------------------------------------------------------------------

    def predict_liquidity(
        self,
        symbol: str,
        horizon_days: int = 5,
    ) -> list[LiquidityResult]:
        """Predict liquidity tier probabilities for the next N trading days.

        Returns P(low), P(medium), P(high) for each future trading day.

        Args:
            symbol: BVMT ticker symbol.
            horizon_days: Number of future trading days (1-5).

        Returns:
            List of LiquidityResult with probability vectors.
        """
        features = self._get_latest_features(symbol)
        if features is None or features.empty:
            logger.warning("No features for liquidity prediction on %s.", symbol)
            return []

        from prediction.etl.transform.silver_to_gold import SilverToGoldTransformer
        feature_cols = SilverToGoldTransformer.get_feature_columns(features)
        X = features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        liq_model = self._load_liquidity_model(symbol)
        if liq_model is None:
            logger.warning("No liquidity model found for %s.", symbol)
            return []

        results: list[LiquidityResult] = []
        base_date = date.today()

        try:
            probas = liq_model.predict_proba_tiers(X)
            # Use the last row's probability for each horizon day
            last_proba = probas[-1] if probas else None

            if last_proba is not None:
                for day_offset in range(1, horizon_days + 1):
                    target_date = self._next_trading_day(base_date, day_offset)
                    results.append(LiquidityResult(
                        symbol=symbol,
                        target_date=target_date,
                        prob_low=last_proba.prob_low,
                        prob_medium=last_proba.prob_medium,
                        prob_high=last_proba.prob_high,
                    ))
        except Exception:
            logger.exception("Liquidity prediction failed for %s", symbol)

        return results

    def _load_liquidity_model(self, symbol: str) -> LiquidityClassifier | None:
        """Load the liquidity classifier from registry."""
        candidates: list[Path] = []
        if symbol:
            candidates.append(self._models_dir / "liquidity" / symbol.upper())
        candidates.append(self._models_dir / "liquidity")

        for path in candidates:
            if path.exists() and (path / "liquidity_xgb_model.json").exists():
                try:
                    model = LiquidityClassifier()
                    model.load_model(path)
                    return model
                except Exception:
                    logger.warning("Failed to load liquidity model from %s", path)
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
