"""
Redis cache client for predictions and feature store.

Implements:
- Prediction cache (TTL: 1 hour intraday, 12h post-market)
- Feature store (TTL: 90 days rolling)
- Event-driven invalidation
- Cache warming for top tickers
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    logger.warning("Redis not installed. Caching will be disabled.")


class CacheClient:
    """Redis-backed cache for predictions and features.

    Falls back to an in-memory dict if Redis is unavailable.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        prediction_ttl: int = 3600,
        post_market_ttl: int = 43200,
        feature_ttl_days: int = 90,
    ) -> None:
        self._prediction_ttl = prediction_ttl
        self._post_market_ttl = post_market_ttl
        self._feature_ttl = timedelta(days=feature_ttl_days)
        self._redis: "redis.Redis | None" = None
        self._memory_cache: dict[str, Any] = {}

        if HAS_REDIS:
            try:
                self._redis = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_timeout=5,
                )
                self._redis.ping()
                logger.info("Connected to Redis at %s", redis_url)
            except Exception:
                logger.warning(
                    "Cannot connect to Redis. Using in-memory cache."
                )
                self._redis = None

    # ------------------------------------------------------------------
    # Prediction Cache
    # ------------------------------------------------------------------

    def get_prediction(self, ticker: str, model: str = "ensemble") -> dict | None:
        """Retrieve a cached prediction.

        Key pattern: pred:{ticker}:{model}
        """
        key = f"pred:{ticker}:{model}"
        return self._get(key)

    def set_prediction(
        self, ticker: str, prediction: dict, model: str = "ensemble"
    ) -> None:
        """Cache a prediction result.

        TTL depends on market hours (1h intraday, 12h post-market).
        """
        key = f"pred:{ticker}:{model}"
        ttl = self._get_prediction_ttl()
        self._set(key, prediction, ttl)

    def invalidate_predictions(self, ticker: str | None = None) -> int:
        """Invalidate cached predictions.

        Args:
            ticker: Specific ticker to invalidate, or None for all.

        Returns:
            Number of keys invalidated.
        """
        pattern = f"pred:{ticker}:*" if ticker else "pred:*"
        return self._delete_pattern(pattern)

    # ------------------------------------------------------------------
    # Feature Store
    # ------------------------------------------------------------------

    def get_features(self, code: str, date_str: str) -> dict | None:
        """Retrieve cached features.

        Key pattern: features:{CODE}:{DATE}
        """
        key = f"features:{code}:{date_str}"
        return self._get(key)

    def set_features(self, code: str, date_str: str, features: dict) -> None:
        """Cache engineered features.

        TTL: 90 days (rolling window).
        """
        key = f"features:{code}:{date_str}"
        ttl = int(self._feature_ttl.total_seconds())
        self._set(key, features, ttl)

    # ------------------------------------------------------------------
    # Cache Warming
    # ------------------------------------------------------------------

    def warm_cache(
        self, tickers: list[str], prediction_fn: "callable"
    ) -> int:
        """Pre-compute and cache predictions for top tickers.

        Called daily at 8:00 AM to ensure fast response for
        the most liquid stocks.

        Args:
            tickers: List of ticker symbols to warm.
            prediction_fn: Callable that generates prediction dict for a ticker.

        Returns:
            Number of tickers warmed.
        """
        warmed = 0
        for ticker in tickers:
            try:
                prediction = prediction_fn(ticker)
                self.set_prediction(ticker, prediction)
                warmed += 1
            except Exception:
                logger.warning("Failed to warm cache for %s", ticker)
        logger.info("Cache warmed for %d/%d tickers.", warmed, len(tickers))
        return warmed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, key: str) -> dict | None:
        """Get a value from cache."""
        if self._redis is not None:
            try:
                val = self._redis.get(key)
                if val is not None:
                    return json.loads(val)
            except Exception:
                logger.warning("Redis GET failed for key %s", key)
        return self._memory_cache.get(key)

    def _set(self, key: str, value: dict, ttl: int) -> None:
        """Set a value in cache with TTL."""
        serialized = json.dumps(value, default=str)
        if self._redis is not None:
            try:
                self._redis.setex(key, ttl, serialized)
                return
            except Exception:
                logger.warning("Redis SET failed for key %s", key)
        self._memory_cache[key] = value

    def _delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern."""
        if self._redis is not None:
            try:
                keys = list(self._redis.scan_iter(match=pattern))
                if keys:
                    self._redis.delete(*keys)
                return len(keys)
            except Exception:
                logger.warning("Redis DELETE failed for pattern %s", pattern)

        # In-memory fallback
        import fnmatch
        to_delete = [k for k in self._memory_cache if fnmatch.fnmatch(k, pattern)]
        for k in to_delete:
            del self._memory_cache[k]
        return len(to_delete)

    def _get_prediction_ttl(self) -> int:
        """Determine TTL based on market hours.

        BVMT: 09:00–14:15 Tunisia time (UTC+1).
        During market hours → 1 hour TTL.
        After market close → 12 hour TTL.
        """
        now = datetime.now()
        hour = now.hour
        if 9 <= hour < 15:
            return self._prediction_ttl
        return self._post_market_ttl
