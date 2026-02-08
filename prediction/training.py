"""
Training pipeline orchestrator.

Implements walk-forward cross-validation for time series:
- Multiple train/val/test splits
- Per-model training + evaluation
- Ensemble weight optimization
- Model persistence via registry
- Anti-leakage enforcement
- **MLflow experiment tracking** for all runs & metrics

Usage:
    pipeline = TrainingPipeline(config)
    pipeline.run(features_df)
"""

import logging
from dataclasses import dataclass, asdict
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

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    logger.warning("MLflow not installed. Experiment tracking disabled.")


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


def _setup_mlflow(cfg: PredictionConfig) -> bool:
    """Initialise MLflow tracking. Returns True if available."""
    if not HAS_MLFLOW:
        return False
    try:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        logger.info(
            "MLflow tracking: uri=%s  experiment=%s",
            cfg.mlflow.tracking_uri,
            cfg.mlflow.experiment_name,
        )
        return True
    except Exception:
        logger.warning("MLflow setup failed. Tracking disabled.")
        return False


def _log_params_for_model(model: BasePredictionModel) -> None:
    """Log hyperparameters of a model to the active MLflow run."""
    if not HAS_MLFLOW:
        return
    if isinstance(model, LSTMPredictor):
        mlflow.log_params({
            "lstm_seq_len": model._seq_len,
            "lstm_hidden": model._hidden_size,
            "lstm_layers": model._num_layers,
            "lstm_dropout": model._dropout,
            "lstm_lr": model._lr,
            "lstm_epochs": model._epochs,
            "lstm_batch_size": model._batch_size,
            "lstm_patience": model._patience,
        })
    elif isinstance(model, XGBoostPredictor):
        mlflow.log_params({
            "xgb_n_estimators": model._n_estimators,
            "xgb_max_depth": model._max_depth,
            "xgb_lr": model._learning_rate,
            "xgb_subsample": model._subsample,
            "xgb_colsample": model._colsample_bytree,
        })
    elif isinstance(model, ProphetPredictor):
        mlflow.log_params({
            "prophet_cp_prior": model._cp_prior,
            "prophet_season_prior": model._season_prior,
            "prophet_yearly": model._yearly,
            "prophet_weekly": model._weekly,
        })


def _log_metrics_to_mlflow(
    model_name: str, metrics: ModelMetrics, split_name: str
) -> None:
    """Log evaluation metrics to the active MLflow run."""
    if not HAS_MLFLOW:
        return
    prefix = f"{split_name}/"
    mlflow.log_metrics({
        f"{prefix}mae": metrics.mae,
        f"{prefix}rmse": metrics.rmse,
        f"{prefix}mape": metrics.mape,
        f"{prefix}directional_accuracy": metrics.directional_accuracy,
        f"{prefix}r_squared": metrics.r_squared,
    })


class TrainingPipeline:
    """End-to-end ML training pipeline with MLflow tracking.

    Steps:
    1. Split data (walk-forward CV)
    2. Train each base model
    3. Evaluate on validation set
    4. Log all params + metrics to MLflow
    5. Optimize ensemble weights
    6. Final evaluation on test set
    7. Persist best models + register in MLflow
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
        self._mlflow_ok = _setup_mlflow(self._cfg)

    def run(
        self,
        features_df: pd.DataFrame,
        target_col: str = "target_1d",
        feature_cols: list[str] | None = None,
        symbol: str | None = None,
    ) -> dict[str, ModelMetrics]:
        """Execute the full training pipeline with MLflow tracking.

        Each model × CV-split combination gets its own MLflow child run.
        A parent run groups the entire training session.

        Args:
            features_df: Gold-layer DataFrame with features + targets.
            target_col: Name of the target column.
            feature_cols: List of feature column names. Auto-detected if None.
            symbol: If provided, models are saved under models/{symbol}/.

        Returns:
            Dict mapping model name → average metrics across CV splits.
        """
        if feature_cols is None:
            gold_transformer = SilverToGoldTransformer()
            feature_cols = gold_transformer.get_feature_columns(features_df)
            feature_cols = [
                c for c in feature_cols
                if not c.startswith("target_") and c != target_col
            ]

        all_metrics: dict[str, list[ModelMetrics]] = {}

        # ── Parent MLflow run for the whole training session ──
        parent_ctx = (
            mlflow.start_run(run_name="walk-forward-cv", nested=False)
            if self._mlflow_ok
            else _nullcontext()
        )

        with parent_ctx:
            if self._mlflow_ok:
                mlflow.log_params({
                    "target_col": target_col,
                    "n_features": len(feature_cols),
                    "n_cv_splits": len(self._cv_splits),
                    "total_rows": len(features_df),
                })

            for split in self._cv_splits:
                logger.info("=" * 60)
                logger.info("Training CV split: %s", split.name)
                logger.info("=" * 60)

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

                X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
                X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)

                models = self._create_models()

                for model in models:
                    logger.info("Training %s on %s...", model.name, split.name)

                    # ── Child MLflow run per model×split ──
                    child_ctx = (
                        mlflow.start_run(
                            run_name=f"{model.name}_{split.name}",
                            nested=True,
                        )
                        if self._mlflow_ok
                        else _nullcontext()
                    )

                    with child_ctx:
                        try:
                            if self._mlflow_ok:
                                mlflow.log_params({
                                    "model": model.name,
                                    "split": split.name,
                                    "train_end_year": split.train_end_year,
                                    "val_end_year": split.val_end_year,
                                    "train_rows": len(X_train),
                                    "val_rows": len(X_val),
                                })
                                _log_params_for_model(model)

                            model.fit(X_train, y_train, X_val, y_val)
                            metrics = model.evaluate(X_val, y_val)

                            if self._mlflow_ok:
                                mlflow.log_metrics({
                                    "mae": metrics.mae,
                                    "rmse": metrics.rmse,
                                    "mape": metrics.mape,
                                    "directional_accuracy": metrics.directional_accuracy,
                                    "r_squared": metrics.r_squared,
                                })

                            if model.name not in all_metrics:
                                all_metrics[model.name] = []
                            all_metrics[model.name].append(metrics)

                        except Exception:
                            logger.exception(
                                "Failed to train/evaluate %s on %s",
                                model.name,
                                split.name,
                            )

            # ── Average metrics → parent run ──
            avg_metrics = self._average_metrics(all_metrics)
            self._log_summary(avg_metrics)

            if self._mlflow_ok:
                for name, m in avg_metrics.items():
                    mlflow.log_metrics({
                        f"avg_{name}_mae": m.mae,
                        f"avg_{name}_rmse": m.rmse,
                        f"avg_{name}_mape": m.mape,
                        f"avg_{name}_dir_acc": m.directional_accuracy,
                        f"avg_{name}_r2": m.r_squared,
                    })

        return avg_metrics

    def train_final_model(
        self,
        features_df: pd.DataFrame,
        target_col: str = "target_1d",
        feature_cols: list[str] | None = None,
        symbol: str | None = None,
    ) -> EnsemblePredictor:
        """Train the final production model on all available data.

        Logged as a dedicated MLflow run tagged 'production'.

        Args:
            features_df: Feature DataFrame (optionally pre-filtered to one symbol).
            target_col: Target column name.
            feature_cols: Feature column names. Auto-detected if None.
            symbol: If provided, saves models under models/ensemble/{symbol}/.

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

        # Final split: train on everything except most recent year for validation
        # Auto-detect split years from the actual data
        max_year = pd.to_datetime(features_df["seance"]).dt.year.max()
        val_year = max_year
        train_year = max_year - 1

        transformer = SilverToGoldTransformer(
            train_end_year=train_year,
            val_end_year=val_year,
        )
        train_df, val_df, _ = transformer.create_training_view(features_df)

        # If val is empty (e.g. data ends exactly at train_year), use last 20%
        if val_df.empty:
            logger.warning(
                "Validation set empty with train_end=%d, val_end=%d. "
                "Using last 20%% of data as validation.",
                train_year, val_year,
            )
            all_df = train_df.sort_values("seance").reset_index(drop=True)
            split_idx = int(len(all_df) * 0.8)
            train_df = all_df.iloc[:split_idx]
            val_df = all_df.iloc[split_idx:]

        logger.info(
            "Final model split — Train: %d rows, Val: %d rows",
            len(train_df), len(val_df),
        )

        X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_train = train_df[target_col]
        X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_val = val_df[target_col]

        models = self._create_models()
        ensemble = EnsemblePredictor(
            models={m.name: m for m in models},
            weights={
                "LSTM": self._cfg.model.ensemble_lstm_weight,
                "XGBoost": self._cfg.model.ensemble_xgb_weight,
                "Prophet": self._cfg.model.ensemble_prophet_weight,
            },
        )

        run_ctx = (
            mlflow.start_run(run_name="final-production-model")
            if self._mlflow_ok
            else _nullcontext()
        )

        with run_ctx:
            if self._mlflow_ok:
                mlflow.set_tag("stage", "production")
                mlflow.log_params({
                    "target_col": target_col,
                    "n_features": len(feature_cols),
                    "train_rows": len(X_train),
                    "val_rows": len(X_val),
                    "lstm_weight": self._cfg.model.ensemble_lstm_weight,
                    "xgb_weight": self._cfg.model.ensemble_xgb_weight,
                    "prophet_weight": self._cfg.model.ensemble_prophet_weight,
                })

            ensemble.fit(X_train, y_train, X_val, y_val)

            # Evaluate each base model and log
            for name, model in ensemble._models.items():
                if model.is_fitted:
                    metrics = model.evaluate(X_val, y_val)
                    logger.info("[Final %s] %s", name, metrics)
                    if self._mlflow_ok:
                        mlflow.log_metrics({
                            f"final_{name}_mae": metrics.mae,
                            f"final_{name}_rmse": metrics.rmse,
                            f"final_{name}_mape": metrics.mape,
                            f"final_{name}_dir_acc": metrics.directional_accuracy,
                            f"final_{name}_r2": metrics.r_squared,
                        })

            # Log ensemble weights after optimization
            if self._mlflow_ok:
                mlflow.log_metrics({
                    f"opt_weight_{k}": v
                    for k, v in ensemble._weights.items()
                })

            # Save to registry
            self._save_models(ensemble, symbol=symbol)

            # Log model artifacts to MLflow
            if self._mlflow_ok:
                mlflow.log_artifacts(
                    str(self._models_dir / "ensemble"), artifact_path="ensemble"
                )

        return ensemble

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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

    def _save_models(self, ensemble: EnsemblePredictor, symbol: str | None = None) -> None:
        """Persist the trained ensemble to the model registry.

        Args:
            ensemble: Trained ensemble predictor.
            symbol: If provided, save under models/ensemble/{symbol}/.
        """
        self._models_dir.mkdir(parents=True, exist_ok=True)
        if symbol:
            save_path = self._models_dir / "ensemble" / symbol.upper()
        else:
            save_path = self._models_dir / "ensemble"
        ensemble.save_model(save_path)
        logger.info("Models saved to %s", save_path)

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


class _nullcontext:
    """Minimal no-op context manager (for when MLflow is disabled)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
