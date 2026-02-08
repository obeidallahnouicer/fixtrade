"""
FixTrade Prediction Module
===========================

Modular ML system for BVMT stock price prediction.

Architecture
------------
- **Data**: Medallion (Bronze → Silver → Gold) with Parquet storage
- **Features**: 50+ engineered indicators (technical, temporal, volume, lag)
- **Models**: LSTM · XGBoost · Prophet → Ensemble (liquidity-tiered)
- **Serving**: Redis-backed cache, walk-forward validated
- **Real-time**: WebSocket/SSE streaming, scheduler, file watcher

Quick start (CLI)
-----------------
    python -m prediction etl               # ingest + transform raw data
    python -m prediction train --final     # train ensemble on all data
    python -m prediction predict --symbol BIAT --days 3
    python -m prediction scheduler         # start automated pipeline
    python -m prediction watch             # watch data/raw for new CSVs
    python -m prediction stream            # start WebSocket/SSE server

Public API
----------
    from prediction import PredictionService, ETLPipeline, TrainingPipeline
    from prediction.realtime import RealtimeScheduler, PredictionStreamManager, DataWatcher
    from prediction.config import config
"""

# ── Public façade ──────────────────────────────────────────────────
from prediction.config import PredictionConfig, config
from prediction.pipeline import ETLPipeline
from prediction.training import TrainingPipeline
from prediction.inference import (
    PredictionService,
    PredictionResult,
    VolumeResult,
    LiquidityResult,
)

__all__ = [
    "PredictionConfig",
    "config",
    "ETLPipeline",
    "TrainingPipeline",
    "PredictionService",
    "PredictionResult",
    "VolumeResult",
    "LiquidityResult",
]
