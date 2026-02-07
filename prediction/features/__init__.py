"""
Feature engineering sub-package.

Generators (each applied per-ticker with .shift(1) anti-leakage)
----------------------------------------------------------------
- `TechnicalFeatures` — SMA, EMA, RSI, MACD, Bollinger, ATR, Stochastic, ROC, OBV
- `TemporalFeatures`  — calendar, cyclical encoding, Tunisian holidays
- `VolumeFeatures`    — volume SMA, VWAP, A/D line, MFI
- `LagFeatures`       — price lags, return lags, rolling stats, momentum, drawdown

Orchestrator
------------
- `FeaturePipeline.run(silver_df)` — applies all generators, returns enriched DataFrame
"""

from prediction.features.pipeline import FeaturePipeline
from prediction.features.technical import TechnicalFeatures
from prediction.features.temporal import TemporalFeatures
from prediction.features.volume import VolumeFeatures
from prediction.features.lag import LagFeatures

__all__ = [
    "FeaturePipeline",
    "TechnicalFeatures",
    "TemporalFeatures",
    "VolumeFeatures",
    "LagFeatures",
]
