"""
Feature pipeline orchestrator.

Runs all feature engineering steps in order:
1. Technical indicators
2. Temporal / calendar features
3. Volume profile features
4. Lag / momentum features

Outputs a fully-featured Silver-layer DataFrame.
"""

import logging

import pandas as pd

from prediction.features.lag import LagFeatures
from prediction.features.technical import TechnicalFeatures
from prediction.features.temporal import TemporalFeatures
from prediction.features.volume import VolumeFeatures
from prediction.config import FeatureConfig, config

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Orchestrates the full feature engineering pipeline.

    Processes each ticker independently to avoid cross-contamination,
    then concatenates the results.
    """

    def __init__(self, cfg: FeatureConfig | None = None) -> None:
        cfg = cfg or config.features
        self._technical = TechnicalFeatures(
            sma_windows=cfg.sma_windows,
            ema_windows=cfg.ema_windows,
            rsi_window=cfg.rsi_window,
            macd_fast=cfg.macd_fast,
            macd_slow=cfg.macd_slow,
            macd_signal=cfg.macd_signal,
            bollinger_window=cfg.bollinger_window,
            bollinger_std=cfg.bollinger_std,
            atr_window=cfg.atr_window,
            stochastic_window=cfg.stochastic_window,
        )
        self._temporal = TemporalFeatures()
        self._volume = VolumeFeatures(
            sma_windows=cfg.volume_sma_windows,
            vwap_window=cfg.vwap_window,
        )
        self._lag = LagFeatures(lag_days=cfg.lag_days)

    def run(self, silver_df: pd.DataFrame) -> pd.DataFrame:
        """Execute the full feature pipeline on Silver-layer data.

        Args:
            silver_df: Cleaned Silver-layer DataFrame with standard columns.

        Returns:
            Enriched DataFrame with 50+ engineered features.
        """
        if silver_df.empty:
            return silver_df

        # Add temporal features (these are global, not per-ticker)
        df = self._temporal.compute(silver_df)

        # Per-ticker features (technical, volume, lag)
        ticker_groups: list[pd.DataFrame] = []
        for code, group in df.groupby("code"):
            group = group.sort_values("seance").reset_index(drop=True)
            group = self._technical.compute(group)
            group = self._volume.compute(group)
            group = self._lag.compute(group)
            ticker_groups.append(group)

        result = pd.concat(ticker_groups, ignore_index=True)
        result = result.sort_values(["code", "seance"]).reset_index(drop=True)

        feature_cols = [
            c for c in result.columns
            if c not in ("seance", "code", "libelle", "volume")
        ]
        logger.info(
            "Feature pipeline complete: %d rows, %d features",
            len(result),
            len(feature_cols),
        )
        return result
