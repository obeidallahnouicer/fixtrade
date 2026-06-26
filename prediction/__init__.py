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
from importlib import import_module

from prediction.config import PredictionConfig, config

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


def __getattr__(name: str):
    """Lazy-load heavyweight ETL and ML classes on first use."""
    exports = {
        "ETLPipeline": ("prediction.pipeline", "ETLPipeline"),
        "TrainingPipeline": ("prediction.training", "TrainingPipeline"),
        "PredictionService": ("prediction.inference", "PredictionService"),
        "PredictionResult": ("prediction.inference", "PredictionResult"),
        "VolumeResult": ("prediction.inference", "VolumeResult"),
        "LiquidityResult": ("prediction.inference", "LiquidityResult"),
    }
    if name not in exports:
        raise AttributeError(name)
    module_name, attribute_name = exports[name]
    return getattr(import_module(module_name), attribute_name)
