"""
Real-time prediction pipeline.

Provides:
- **RealtimeScheduler**: APScheduler-based task orchestrator for
  periodic ETL, model retraining, and cache warming.
- **PredictionStreamManager**: WebSocket/SSE broadcast of live
  prediction updates to connected clients.
- **DataWatcher**: Filesystem observer that detects new CSV drops
  in data/raw/ and triggers incremental ETL → retrain → broadcast.
"""

from prediction.realtime.scheduler import RealtimeScheduler
from prediction.realtime.stream import PredictionStreamManager
from prediction.realtime.watcher import DataWatcher

__all__ = [
    "RealtimeScheduler",
    "PredictionStreamManager",
    "DataWatcher",
]
