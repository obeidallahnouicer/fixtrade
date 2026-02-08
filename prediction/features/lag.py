"""
Lag features.

Creates autoregressive and return-based features:
- Price lags (shifted closing prices)
- Return lags
- Rolling statistics (mean, std, skew)
- Momentum features
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LagFeatures:
    """Creates time-lagged features from price data.

    All features are inherently lagged (they use past data),
    but an additional .shift(1) is applied where necessary
    to ensure strict temporal separation.
    """

    def __init__(
        self,
        lag_days: tuple[int, ...] = (1, 2, 3, 5, 10, 20),
    ) -> None:
        self._lag_days = lag_days

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute lag and momentum features.

        Args:
            df: Per-ticker DataFrame sorted by date.

        Returns:
            DataFrame with lag feature columns added.
        """
        df = df.copy()
        close = df["cloture"]
        returns = close.pct_change()

        # Price lags
        for lag in self._lag_days:
            df[f"close_lag_{lag}"] = close.shift(lag)
            df[f"return_lag_{lag}"] = returns.shift(lag)

        # Rolling return statistics
        for window in (5, 10, 20):
            df[f"return_mean_{window}d"] = returns.rolling(window).mean().shift(1)
            df[f"return_std_{window}d"] = returns.rolling(window).std().shift(1)
            df[f"return_skew_{window}d"] = returns.rolling(window).skew().shift(1)

        # Cumulative returns over different windows
        for window in (5, 10, 20, 60):
            df[f"cum_return_{window}d"] = (
                (1 + returns).rolling(window).apply(np.prod, raw=True) - 1
            ).shift(1)

        # Max drawdown (20-day rolling)
        rolling_max = close.rolling(20).max()
        drawdown = (close - rolling_max) / rolling_max.replace(0, np.nan)
        df["max_drawdown_20d"] = drawdown.rolling(20).min().shift(1)

        # Momentum (close vs close N days ago)
        for lag in (5, 10, 20):
            df[f"momentum_{lag}d"] = (
                close / close.shift(lag).replace(0, np.nan) - 1
            ).shift(1)

        # Mean reversion signal (z-score of price relative to 20-day SMA)
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        df["mean_reversion_z"] = (
            (close - sma_20) / std_20.replace(0, np.nan)
        ).shift(1)

        logger.debug("Computed lag features with %d lag values.", len(self._lag_days))
        return df
