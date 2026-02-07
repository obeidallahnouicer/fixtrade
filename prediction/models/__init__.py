"""
ML models sub-package.

Strategy Pattern — every model implements `BasePredictionModel`:
    fit(X, y) → predict(X) → evaluate(X, y) → save/load

Concrete models
---------------
- `LSTMPredictor`     — stacked LSTM (PyTorch), best for temporal dependencies
- `XGBoostPredictor`  — gradient-boosted trees, best for feature interactions
- `ProphetPredictor`  — trend + seasonality decomposition, best for low-liquidity

Ensemble
--------
- `EnsemblePredictor` — weighted combination with 3 phases:
    Phase 1: static weights (0.45 / 0.35 / 0.20)
    Phase 2: dynamic weights (volatility + volume aware)
    Phase 3: stacking meta-learner (planned)
"""

from prediction.models.base import BasePredictionModel, ModelMetrics
from prediction.models.lstm import LSTMPredictor
from prediction.models.xgboost_model import XGBoostPredictor
from prediction.models.prophet_model import ProphetPredictor
from prediction.models.ensemble import EnsemblePredictor, EnsemblePrediction, LiquidityTier

__all__ = [
    "BasePredictionModel",
    "ModelMetrics",
    "LSTMPredictor",
    "XGBoostPredictor",
    "ProphetPredictor",
    "EnsemblePredictor",
    "EnsemblePrediction",
    "LiquidityTier",
]
