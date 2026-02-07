"""
Utilities sub-package.

- `CacheClient`   — Redis + in-memory fallback, TTL-aware prediction/feature cache
- `ModelMonitor`  — rolling performance tracking, drift detection, retrain alerts
"""

from prediction.utils.cache import CacheClient
from prediction.utils.metrics import ModelMonitor, PerformanceReport

__all__ = [
    "CacheClient",
    "ModelMonitor",
    "PerformanceReport",
]
