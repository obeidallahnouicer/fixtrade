"""
Transform layer: Silver → Gold ML-ready views.

Produces three Gold-layer views:
- Training dataset (features + labels)
- Inference dataset (latest features, no labels)
- Evaluation dataset (holdout test set, time-series split)

Targets include:
- Price targets: target_{h}d  (future closing price at horizon h)
- Volume targets: target_volume_{h}d  (future daily volume at horizon h)
- Liquidity labels: liquidity_label  (0=low, 1=medium, 2=high)
"""

import logging
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Liquidity tier thresholds (match LiquidityTierConfig defaults) ───
_HIGH_VOL_THRESHOLD = 10_000
_MED_VOL_THRESHOLD = 1_000


class SilverToGoldTransformer:
    """Transforms enriched Silver data into ML-ready Gold datasets.

    Gold datasets include targets (next 1–5 day closing prices),
    volume targets (next 1–5 day volumes), liquidity labels,
    and strict chronological train/validation/test splits.
    """

    def __init__(
        self,
        train_end_year: int = 2024,
        val_end_year: int = 2025,
        prediction_horizons: tuple[int, ...] = (1, 2, 3, 5),
        volume_horizons: tuple[int, ...] = (1, 2, 3, 5),
    ) -> None:
        self._train_end = train_end_year
        self._val_end = val_end_year
        self._horizons = prediction_horizons
        self._vol_horizons = volume_horizons

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

        # Drop rows where PRICE targets are NaN (end of each ticker's data).
        # Volume/liquidity targets have their own NaN handling in training.
        price_target_cols = [
            c for c in df.columns
            if c.startswith("target_") and "volume" not in c
        ]
        df = df.dropna(subset=price_target_cols)

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

        Creates:
        - target_{h}d: closing price h days ahead
        - target_volume_{h}d: volume h days ahead
        - liquidity_label: 0=low, 1=medium, 2=high (based on next-day volume)

        Uses .shift(-h) within each ticker group to prevent leakage.
        """
        df = df.copy()

        vol_col = self._detect_volume_column(df)

        # Price targets
        for h in self._horizons:
            col_name = f"target_{h}d"
            df[col_name] = df.groupby("code")["cloture"].shift(-h)

        # Volume targets
        if vol_col:
            for h in self._vol_horizons:
                col_name = f"target_volume_{h}d"
                df[col_name] = df.groupby("code")[vol_col].shift(-h)

            # Liquidity label based on next-day volume
            next_vol = df.groupby("code")[vol_col].shift(-1)
            df["liquidity_label"] = self._volume_to_liquidity_label(next_vol)

        return df

    @staticmethod
    def _detect_volume_column(df: pd.DataFrame) -> str | None:
        """Find the volume column in the DataFrame."""
        for candidate in ("quantite_negociee", "volume", "Volume"):
            if candidate in df.columns:
                return candidate
        return None

    @staticmethod
    def _volume_to_liquidity_label(volume: pd.Series) -> pd.Series:
        """Convert volume series to integer liquidity tier labels.

        0 = low  (volume < 1,000)
        1 = medium (1,000 ≤ volume < 10,000)
        2 = high (volume ≥ 10,000)
        """
        labels = pd.Series(np.nan, index=volume.index, dtype="float64")
        labels[volume >= _HIGH_VOL_THRESHOLD] = 2
        labels[(volume >= _MED_VOL_THRESHOLD) & (volume < _HIGH_VOL_THRESHOLD)] = 1
        labels[volume < _MED_VOL_THRESHOLD] = 0
        # Keep NaN where volume is NaN (end of each ticker's data)
        labels[volume.isna()] = np.nan
        return labels

    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> list[str]:
        """Return the list of feature column names (exclude targets, metadata)."""
        exclude_prefixes = (
            "target_", "liquidity_label",
            "seance", "code", "libelle", "year", "month",
        )
        return [
            c
            for c in df.columns
            if not any(c.startswith(p) for p in exclude_prefixes)
        ]
