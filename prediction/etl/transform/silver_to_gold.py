"""
Transform layer: Silver → Gold ML-ready views.

Produces three Gold-layer views:
- Training dataset (features + labels)
- Inference dataset (latest features, no labels)
- Evaluation dataset (holdout test set, time-series split)
"""

import logging
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SilverToGoldTransformer:
    """Transforms enriched Silver data into ML-ready Gold datasets.

    Gold datasets include targets (next 1–5 day closing prices)
    and strict chronological train/validation/test splits.
    """

    def __init__(
        self,
        train_end_year: int = 2024,
        val_end_year: int = 2025,
        prediction_horizons: tuple[int, ...] = (1, 2, 3, 5),
    ) -> None:
        self._train_end = train_end_year
        self._val_end = val_end_year
        self._horizons = prediction_horizons

    def create_training_view(
        self, features_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train / validation / test splits (walk-forward).

        Anti-leakage guarantee:
        - Train < Validation < Test chronologically.
        - No random shuffling.

        Args:
            features_df: Silver-layer DataFrame with features and 'cloture' column.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        df = features_df.copy()
        df = self._add_targets(df)

        # Drop rows where targets are NaN (end of each ticker's data)
        target_cols = [c for c in df.columns if c.startswith("target_")]
        df = df.dropna(subset=target_cols)

        year = df["seance"].dt.year

        train = df[year <= self._train_end].reset_index(drop=True)
        val = df[
            (year > self._train_end) & (year <= self._val_end)
        ].reset_index(drop=True)
        test = df[year > self._val_end].reset_index(drop=True)

        logger.info(
            "Gold splits — Train: %d, Val: %d, Test: %d",
            len(train),
            len(val),
            len(test),
        )
        return train, val, test

    def create_inference_view(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Create the inference dataset: latest features per ticker.

        No target columns — used at prediction time.
        """
        df = features_df.copy()
        # Keep only the most recent row per ticker
        latest = df.sort_values("seance").groupby("code").tail(1)
        return latest.reset_index(drop=True)

    def create_evaluation_view(
        self, features_df: pd.DataFrame, holdout_start: date | None = None
    ) -> pd.DataFrame:
        """Create an evaluation holdout set for backtesting.

        Args:
            features_df: Full feature DataFrame.
            holdout_start: Start date for holdout period.

        Returns:
            Holdout DataFrame with features and targets.
        """
        df = features_df.copy()
        df = self._add_targets(df)

        if holdout_start is None:
            holdout_start = date(self._val_end, 7, 1)

        target_cols = [c for c in df.columns if c.startswith("target_")]
        df = df.dropna(subset=target_cols)

        mask = df["seance"].dt.date >= holdout_start
        return df[mask].reset_index(drop=True)

    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add forward-looking target columns per ticker.

        For each horizon h, target_h = closing price h days ahead.
        Uses .shift(-h) within each ticker group to prevent leakage.
        """
        df = df.copy()
        for h in self._horizons:
            col_name = f"target_{h}d"
            df[col_name] = df.groupby("code")["cloture"].shift(-h)
        return df

    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> list[str]:
        """Return the list of feature column names (exclude targets, metadata)."""
        exclude_prefixes = ("target_", "seance", "code", "libelle", "year", "month")
        return [
            c
            for c in df.columns
            if not any(c.startswith(p) for p in exclude_prefixes)
        ]
