"""
Volume profile features.

Computes volume-based indicators:
- Volume SMA / EMA
- VWAP (Volume-Weighted Average Price)
- Volume ratio (current vs average)
- Volume trend (increasing/decreasing)
- Accumulation/Distribution Line
- Money Flow Index (MFI)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VolumeFeatures:
    """Generates volume-based features from OHLCV data.

    All features are shifted by 1 day to prevent look-ahead bias.
    """

    def __init__(
        self,
        sma_windows: tuple[int, ...] = (5, 10, 20),
        vwap_window: int = 20,
    ) -> None:
        self._sma_windows = sma_windows
        self._vwap_window = vwap_window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume features for a per-ticker DataFrame.

        Args:
            df: OHLCV data for a single ticker, sorted by date.

        Returns:
            DataFrame with additional volume feature columns.
        """
        df = df.copy()
        volume = df["quantite_negociee"].astype(float)
        close = df["cloture"]
        high = df["plus_haut"]
        low = df["plus_bas"]

        # Volume Moving Averages
        for w in self._sma_windows:
            df[f"volume_sma_{w}"] = volume.rolling(window=w).mean().shift(1)

        # Volume Ratio (current volume vs 20-day average)
        vol_avg = volume.rolling(window=20).mean()
        df["volume_ratio"] = (volume / vol_avg.replace(0, np.nan)).shift(1)

        # Volume trend (5-day slope)
        df["volume_trend"] = (
            volume.rolling(5).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0,
                raw=True,
            ).shift(1)
        )

        # VWAP (Volume-Weighted Average Price)
        typical_price = (close + high + low) / 3
        cum_tp_vol = (typical_price * volume).rolling(window=self._vwap_window).sum()
        cum_vol = volume.rolling(window=self._vwap_window).sum()
        df["vwap"] = (cum_tp_vol / cum_vol.replace(0, np.nan)).shift(1)

        # Price relative to VWAP
        df["price_to_vwap"] = (close / df["vwap"].replace(0, np.nan) - 1).shift(1)

        # Accumulation/Distribution Line
        mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
        ad_line = (mfm * volume).fillna(0).cumsum()
        df["ad_line"] = ad_line.shift(1)

        # Money Flow Index (MFI)
        df["mfi"] = self._compute_mfi(
            high, low, close, volume, window=14
        ).shift(1)

        logger.debug("Computed volume features.")
        return df

    @staticmethod
    def _compute_mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        window: int = 14,
    ) -> pd.Series:
        """Compute Money Flow Index."""
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        pos_sum = positive_flow.rolling(window).sum()
        neg_sum = negative_flow.rolling(window).sum()
        mfr = pos_sum / neg_sum.replace(0, np.nan)
        return 100 - (100 / (1 + mfr))
