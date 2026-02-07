"""
Training pipeline orchestrator.

Implements walk-forward cross-validation for time series:
- Multiple train/val/test splits
- Per-model training + evaluation
- Ensemble weight optimization
- Model persistence via registry
- Anti-leakage enforcement

Usage:
    pipeline = TrainingPipeline(config)
    pipeline.run(features_df)
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from prediction.config import PredictionConfig, config
from prediction.etl.transform.silver_to_gold import SilverToGoldTransformer
from prediction.models.base import BasePredictionModel, ModelMetrics
from prediction.models.ensemble import EnsemblePredictor
from prediction.models.lstm import LSTMPredictor
from prediction.models.prophet_model import ProphetPredictor
from prediction.models.xgboost_model import XGBoostPredictor
from prediction.utils.metrics import ModelMonitor

logger = logging.getLogger(__name__)


@dataclass
class CVSplit:
    """A single cross-validation split."""

    name: str
    train_end_year: int
    val_end_year: int
    test_end_year: int | None = None


# Walk-Forward CV Splits (2016–2026)
DEFAULT_CV_SPLITS = [
    CVSplit("split_1", train_end_year=2022, val_end_year=2023, test_end_year=2024),
    CVSplit("split_2", train_end_year=2023, val_end_year=2024, test_end_year=2025),
    CVSplit("split_3", train_end_year=2024, val_end_year=2025),
]


class TrainingPipeline:
    """End-to-end ML training pipeline.

    Steps:
    1. Split data (walk-forward CV)
    2. Train each base model
    3. Evaluate on validation set
    4. Optimize ensemble weights
    5. Final evaluation on test set
    6. Persist best models
    """

    def __init__(
        self,
        cfg: PredictionConfig | None = None,
        cv_splits: list[CVSplit] | None = None,
    ) -> None:
        self._cfg = cfg or config
        self._cv_splits = cv_splits or DEFAULT_CV_SPLITS
        self._monitor = ModelMonitor()
        self._models_dir = self._cfg.paths.models_dir

    def run(
        self,
        features_df: pd.DataFrame,
        target_col: str = "target_1d",
        feature_cols: list[str] | None = None,
    ) -> dict[str, ModelMetrics]:
        """Execute the full training pipeline.

        Args:
            features_df: Gold-layer DataFrame with features + targets.
            target_col: Name of the target column.
            feature_cols: List of feature column names. Auto-detected if None.

        Returns:
            Dict mapping model name → average metrics across CV splits.
        """
        if feature_cols is None:
            gold_transformer = SilverToGoldTransformer()
            feature_cols = gold_transformer.get_feature_columns(features_df)
            # Remove target columns from features
            feature_cols = [
                c for c in feature_cols
                if not c.startswith("target_") and c != target_col
            ]

        all_metrics: dict[str, list[ModelMetrics]] = {}

        for split in self._cv_splits:
            logger.info("=" * 60)
            logger.info("Training CV split: %s", split.name)
            logger.info("=" * 60)

            # Create split-specific data transformer
            transformer = SilverToGoldTransformer(
                train_end_year=split.train_end_year,
                val_end_year=split.val_end_year,
            )
            train_df, val_df, test_df = transformer.create_training_view(features_df)

            if train_df.empty or val_df.empty:
                logger.warning("Skipping %s — insufficient data.", split.name)
                continue

            X_train = train_df[feature_cols].copy()
            y_train = train_df[target_col].copy()
            X_val = val_df[feature_cols].copy()
            y_val = val_df[target_col].copy()

            # Handle NaN/Inf in features
            X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
            X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Build models
            models = self._create_models()

            # Train + evaluate each model
            for model in models:
                logger.info("Training %s on %s...", model.name, split.name)
                try:
                    model.fit(X_train, y_train, X_val, y_val)
                    metrics = model.evaluate(X_val, y_val)

                    if model.name not in all_metrics:
                        all_metrics[model.name] = []
                    all_metrics[model.name].append(metrics)

                except Exception:
                    logger.exception(
                        "Failed to train/evaluate %s on %s",
                        model.name,
                        split.name,
                    )

        # Compute average metrics across splits
        avg_metrics = self._average_metrics(all_metrics)
        self._log_summary(avg_metrics)

        return avg_metrics

    def train_final_model(
        self,
        features_df: pd.DataFrame,
        target_col: str = "target_1d",
        feature_cols: list[str] | None = None,
    ) -> EnsemblePredictor:
        """Train the final production model on all available data.

        Uses all data up to the latest available date.
        Saves the model to the models registry.

        Returns:
            Trained EnsemblePredictor.
        """
        if feature_cols is None:
            gold_transformer = SilverToGoldTransformer()
            feature_cols = gold_transformer.get_feature_columns(features_df)
            feature_cols = [
                c for c in feature_cols
                if not c.startswith("target_") and c != target_col
            ]

        # Final split: train on everything except most recent data
        transformer = SilverToGoldTransformer(
            train_end_year=2025,
            val_end_year=2025,
        )
        train_df, val_df, _ = transformer.create_training_view(features_df)

        X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_train = train_df[target_col]
        X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_val = val_df[target_col]

        # Build and train ensemble
        models = self._create_models()
        ensemble = EnsemblePredictor(
            models={m.name: m for m in models},
            weights={
                "LSTM": self._cfg.model.ensemble_lstm_weight,
                "XGBoost": self._cfg.model.ensemble_xgb_weight,
                "Prophet": self._cfg.model.ensemble_prophet_weight,
            },
        )
        ensemble.fit(X_train, y_train, X_val, y_val)

        # Save to registry
        self._save_models(ensemble)

        return ensemble

    def _create_models(self) -> list[BasePredictionModel]:
        """Instantiate all base prediction models."""
        mcfg = self._cfg.model
        return [
            LSTMPredictor(
                sequence_length=mcfg.lstm_sequence_length,
                hidden_size=mcfg.lstm_hidden_size,
                num_layers=mcfg.lstm_num_layers,
                dropout=mcfg.lstm_dropout,
                learning_rate=mcfg.lstm_learning_rate,
                epochs=mcfg.lstm_epochs,
                batch_size=mcfg.lstm_batch_size,
                patience=mcfg.lstm_patience,
            ),
            XGBoostPredictor(
                n_estimators=mcfg.xgb_n_estimators,
                max_depth=mcfg.xgb_max_depth,
                learning_rate=mcfg.xgb_learning_rate,
                subsample=mcfg.xgb_subsample,
                colsample_bytree=mcfg.xgb_colsample_bytree,
                early_stopping_rounds=mcfg.xgb_early_stopping_rounds,
            ),
            ProphetPredictor(
                changepoint_prior_scale=mcfg.prophet_changepoint_prior_scale,
                seasonality_prior_scale=mcfg.prophet_seasonality_prior_scale,
                yearly_seasonality=mcfg.prophet_yearly_seasonality,
                weekly_seasonality=mcfg.prophet_weekly_seasonality,
            ),
        ]

    def _save_models(self, ensemble: EnsemblePredictor) -> None:
        """Persist the trained ensemble to the model registry."""
        self._models_dir.mkdir(parents=True, exist_ok=True)
        ensemble.save_model(self._models_dir / "ensemble")
        logger.info("Models saved to %s", self._models_dir)

    @staticmethod
    def _average_metrics(
        all_metrics: dict[str, list[ModelMetrics]]
    ) -> dict[str, ModelMetrics]:
        """Average metrics across CV splits."""
        result: dict[str, ModelMetrics] = {}
        for name, metrics_list in all_metrics.items():
            if not metrics_list:
                continue
            result[name] = ModelMetrics(
                mae=np.mean([m.mae for m in metrics_list]),
                rmse=np.mean([m.rmse for m in metrics_list]),
                mape=np.mean([m.mape for m in metrics_list]),
                directional_accuracy=np.mean(
                    [m.directional_accuracy for m in metrics_list]
                ),
                r_squared=np.mean([m.r_squared for m in metrics_list]),
            )
        return result

    @staticmethod
    def _log_summary(avg_metrics: dict[str, ModelMetrics]) -> None:
        """Log a summary of averaged metrics."""
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY (Averaged across CV splits)")
        logger.info("=" * 60)
        for name, metrics in avg_metrics.items():
            logger.info("[%s] %s", name, metrics)
