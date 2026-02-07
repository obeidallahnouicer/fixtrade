"""
Technical indicator features.

Computes 20+ technical analysis indicators from OHLCV data:
- Moving averages (SMA, EMA)
- RSI, MACD, Bollinger Bands
- Stochastic Oscillator, ATR
- Rate of Change, OBV

All indicators use .shift(1) to prevent look-ahead bias.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalFeatures:
    """Computes technical analysis features per ticker.

    ANTI-LEAKAGE: All features use .shift(1) — computed from data
    available BEFORE the current trading day.
    """

    def __init__(
        self,
        sma_windows: tuple[int, ...] = (5, 10, 20, 50, 200),
        ema_windows: tuple[int, ...] = (12, 26),
        rsi_window: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bollinger_window: int = 20,
        bollinger_std: float = 2.0,
        atr_window: int = 14,
        stochastic_window: int = 14,
    ) -> None:
        self._sma_windows = sma_windows
        self._ema_windows = ema_windows
        self._rsi_window = rsi_window
        self._macd_fast = macd_fast
        self._macd_slow = macd_slow
        self._macd_signal = macd_signal
        self._boll_window = bollinger_window
        self._boll_std = bollinger_std
        self._atr_window = atr_window
        self._stoch_window = stochastic_window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical features for a per-ticker DataFrame.

        Expects columns: cloture, ouverture, plus_haut, plus_bas, quantite_negociee.
        All computed columns are shifted by 1 to avoid look-ahead bias.

        Args:
            df: Price data for a single ticker, sorted by seance ascending.

        Returns:
            DataFrame with original columns plus all technical features.
        """
        df = df.copy()
        close = df["cloture"]
        high = df["plus_haut"]
        low = df["plus_bas"]
        volume = df["quantite_negociee"].astype(float)

        # Simple Moving Averages
        for w in self._sma_windows:
            df[f"sma_{w}"] = close.rolling(window=w).mean().shift(1)

        # Exponential Moving Averages
        for w in self._ema_windows:
            df[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean().shift(1)

        # RSI
        df["rsi"] = self._compute_rsi(close, self._rsi_window).shift(1)

        # MACD
        ema_fast = close.ewm(span=self._macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self._macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self._macd_signal, adjust=False).mean()
        df["macd"] = macd_line.shift(1)
        df["macd_signal"] = signal_line.shift(1)
        df["macd_histogram"] = (macd_line - signal_line).shift(1)

        # Bollinger Bands
        sma_boll = close.rolling(window=self._boll_window).mean()
        std_boll = close.rolling(window=self._boll_window).std()
        df["bollinger_upper"] = (sma_boll + self._boll_std * std_boll).shift(1)
        df["bollinger_lower"] = (sma_boll - self._boll_std * std_boll).shift(1)
        df["bollinger_width"] = (
            (df["bollinger_upper"] - df["bollinger_lower"]) / sma_boll.shift(1)
        )

        # ATR (Average True Range)
        df["atr"] = self._compute_atr(high, low, close, self._atr_window).shift(1)

        # Stochastic Oscillator
        lowest_low = low.rolling(window=self._stoch_window).min()
        highest_high = high.rolling(window=self._stoch_window).max()
        denom = highest_high - lowest_low
        df["stochastic_k"] = (
            ((close - lowest_low) / denom.replace(0, np.nan)) * 100
        ).shift(1)
        df["stochastic_d"] = df["stochastic_k"].rolling(3).mean()

        # Rate of Change
        df["roc_5"] = close.pct_change(5).shift(1)
        df["roc_10"] = close.pct_change(10).shift(1)
        df["roc_20"] = close.pct_change(20).shift(1)

        # On-Balance Volume
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df["obv"] = obv.shift(1)

        # Price relative to moving averages
        for w in self._sma_windows:
            sma_col = f"sma_{w}"
            if sma_col in df.columns:
                df[f"price_to_sma_{w}"] = (close / df[sma_col].replace(0, np.nan) - 1).shift(1)

        # Volatility (20-day rolling std of returns)
        returns = close.pct_change()
        df["volatility_20d"] = returns.rolling(20).std().shift(1)

        logger.debug("Computed %d technical features.", len([c for c in df.columns if c not in ("cloture", "ouverture", "plus_haut", "plus_bas", "quantite_negociee", "seance", "code", "libelle", "volume")]))
        return df

    @staticmethod
    def _compute_rsi(series: pd.Series, window: int) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, window: int
    ) -> pd.Series:
        """Compute Average True Range."""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
