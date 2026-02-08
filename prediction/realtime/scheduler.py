"""
Real-time scheduler for automated ETL, retraining, and prediction refresh.

Uses APScheduler to run periodic tasks aligned with BVMT trading hours:
- **Pre-market (08:00)**: Warm prediction cache for top tickers
- **Post-market (15:00)**: Run incremental ETL for new day's data
- **Weekly (Sunday 02:00)**: Full model retraining with latest data
- **On-demand**: Trigger via CLI or API

The scheduler can optionally broadcast events to connected
WebSocket clients via the PredictionStreamManager.

All times are Tunisia (UTC+1 / Africa/Tunis).
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    HAS_APSCHEDULER = True
except ImportError:
    HAS_APSCHEDULER = False
    logger.info("APScheduler not installed. Using built-in threading scheduler.")


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskResult:
    """Result of a scheduled task execution."""

    task_name: str
    status: TaskStatus
    started_at: str
    finished_at: str | None = None
    duration_seconds: float = 0.0
    details: dict = field(default_factory=dict)
    error: str | None = None


class RealtimeScheduler:
    """Orchestrates periodic prediction pipeline tasks.

    Supports two backends:
    - APScheduler (preferred) — cron-like scheduling with misfire handling
    - Built-in threading.Timer (fallback) — simple interval-based scheduling

    All task functions run synchronously in background threads.
    If a PredictionStreamManager is provided, task results are
    broadcast to connected WebSocket clients.

    Usage:
        scheduler = RealtimeScheduler()
        scheduler.start()        # begin all scheduled tasks
        scheduler.stop()         # graceful shutdown
        scheduler.run_now("etl") # trigger a task immediately
    """

    def __init__(
        self,
        stream_manager: Any | None = None,
        top_n_tickers: int = 10,
    ) -> None:
        self._stream = stream_manager
        self._top_n = top_n_tickers
        self._running = False
        self._task_history: list[TaskResult] = []
        self._max_history = 200
        self._lock = threading.Lock()
        self._background_tasks: set[asyncio.Task] = set()  # prevent GC

        # APScheduler backend
        self._scheduler: Any | None = None

        # Fallback threading timers
        self._timers: dict[str, threading.Timer] = {}

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def task_history(self) -> list[TaskResult]:
        return list(self._task_history)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the scheduler with all configured jobs."""
        if self._running:
            logger.warning("Scheduler already running.")
            return

        self._running = True

        if HAS_APSCHEDULER:
            self._start_apscheduler()
        else:
            self._start_fallback()

        logger.info("RealtimeScheduler started.")

    def stop(self) -> None:
        """Gracefully stop the scheduler."""
        self._running = False

        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None

        for name, timer in self._timers.items():
            timer.cancel()
        self._timers.clear()

        logger.info("RealtimeScheduler stopped.")

    def run_now(self, task_name: str) -> TaskResult:
        """Execute a named task immediately (blocking).

        Args:
            task_name: One of 'etl', 'retrain', 'warm_cache',
                       'refresh_predictions', 'full_pipeline'.

        Returns:
            TaskResult with execution details.
        """
        task_map = {
            "etl": self._task_incremental_etl,
            "retrain": self._task_retrain_models,
            "warm_cache": self._task_warm_cache,
            "refresh_predictions": self._task_refresh_predictions,
            "full_pipeline": self._task_full_pipeline,
        }
        fn = task_map.get(task_name)
        if fn is None:
            return TaskResult(
                task_name=task_name,
                status=TaskStatus.FAILED,
                started_at=datetime.now(timezone.utc).isoformat(),
                error=f"Unknown task: {task_name}. "
                      f"Available: {list(task_map.keys())}",
            )
        return fn()

    # ------------------------------------------------------------------
    # APScheduler backend
    # ------------------------------------------------------------------

    def _start_apscheduler(self) -> None:
        """Configure and start APScheduler with cron triggers."""
        self._scheduler = BackgroundScheduler(
            timezone="Africa/Tunis",
            job_defaults={"coalesce": True, "max_instances": 1},
        )

        # Pre-market cache warming: 08:00 Mon–Fri
        self._scheduler.add_job(
            self._task_warm_cache,
            CronTrigger(day_of_week="mon-fri", hour=8, minute=0),
            id="warm_cache",
            name="Pre-market cache warming",
        )

        # Post-market ETL: 15:30 Mon–Fri (after BVMT close at 14:15)
        self._scheduler.add_job(
            self._task_incremental_etl,
            CronTrigger(day_of_week="mon-fri", hour=15, minute=30),
            id="post_market_etl",
            name="Post-market incremental ETL",
        )

        # Post-market prediction refresh: 16:00 Mon–Fri
        self._scheduler.add_job(
            self._task_refresh_predictions,
            CronTrigger(day_of_week="mon-fri", hour=16, minute=0),
            id="refresh_predictions",
            name="Post-market prediction refresh",
        )

        # Weekly full retrain: Sunday 02:00
        self._scheduler.add_job(
            self._task_retrain_models,
            CronTrigger(day_of_week="sun", hour=2, minute=0),
            id="weekly_retrain",
            name="Weekly model retraining",
        )

        # Drift check: every 6 hours
        self._scheduler.add_job(
            self._task_drift_check,
            IntervalTrigger(hours=6),
            id="drift_check",
            name="Model drift detection",
        )

        self._scheduler.start()
        logger.info("APScheduler started with %d jobs.", len(self._scheduler.get_jobs()))

    # ------------------------------------------------------------------
    # Fallback threading scheduler
    # ------------------------------------------------------------------

    def _start_fallback(self) -> None:
        """Start a simple interval-based scheduler using threading."""
        logger.info("Using fallback threading scheduler.")

        # Warm cache every 60 minutes
        self._schedule_interval("warm_cache", self._task_warm_cache, 3600)
        # ETL every 4 hours
        self._schedule_interval("etl", self._task_incremental_etl, 14400)
        # Prediction refresh every 2 hours
        self._schedule_interval("refresh", self._task_refresh_predictions, 7200)
        # Retrain weekly (604800 seconds)
        self._schedule_interval("retrain", self._task_retrain_models, 604800)

    def _schedule_interval(
        self, name: str, fn: Callable, interval_sec: int
    ) -> None:
        """Schedule a function to repeat at a fixed interval."""
        def _loop():
            while self._running:
                fn()
                time.sleep(interval_sec)

        t = threading.Thread(target=_loop, name=f"sched-{name}", daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Task implementations
    # ------------------------------------------------------------------

    def _record_result(self, result: TaskResult) -> None:
        with self._lock:
            self._task_history.append(result)
            if len(self._task_history) > self._max_history:
                self._task_history = self._task_history[-self._max_history:]

        # Broadcast to WebSocket clients if available
        if self._stream is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    task = asyncio.ensure_future(
                        self._stream.broadcast_system_event(
                            f"task_{result.status.value}",
                            {
                                "task": result.task_name,
                                "status": result.status.value,
                                "duration_seconds": result.duration_seconds,
                                "details": result.details,
                                "error": result.error,
                            },
                        )
                    )
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
            except RuntimeError:
                pass  # No running event loop

    def _task_incremental_etl(self) -> TaskResult:
        """Run incremental ETL to ingest new data."""
        start = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            from prediction.pipeline import ETLPipeline

            pipeline = ETLPipeline()
            result = pipeline.run_incremental()
            duration = time.monotonic() - start

            task_result = TaskResult(
                task_name="etl",
                status=TaskStatus.COMPLETED,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_seconds=round(duration, 2),
                details={"rows_processed": len(result) if result is not None else 0},
            )
        except Exception as exc:
            duration = time.monotonic() - start
            task_result = TaskResult(
                task_name="etl",
                status=TaskStatus.FAILED,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_seconds=round(duration, 2),
                error=str(exc),
            )
            logger.exception("Scheduled ETL failed.")

        self._record_result(task_result)
        return task_result

    def _task_retrain_models(self) -> TaskResult:
        """Retrain all models on the latest data."""
        start = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            from prediction.pipeline import ETLPipeline
            from prediction.training import TrainingPipeline

            etl = ETLPipeline()
            features_df = etl._loader.load_layer("silver")

            if features_df.empty:
                raise RuntimeError("No Silver-layer data for retraining.")

            # Filter to top N tickers for speed
            if "code" in features_df.columns and self._top_n:
                counts = features_df["code"].value_counts()
                top_tickers = counts.head(self._top_n).index.tolist()
                features_df = features_df[features_df["code"].isin(top_tickers)]

            trainer = TrainingPipeline()
            ensemble = trainer.train_final_model(features_df)
            duration = time.monotonic() - start

            task_result = TaskResult(
                task_name="retrain",
                status=TaskStatus.COMPLETED,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_seconds=round(duration, 2),
                details={
                    "rows_used": len(features_df),
                    "model": ensemble.name if hasattr(ensemble, "name") else "ensemble",
                },
            )
        except Exception as exc:
            duration = time.monotonic() - start
            task_result = TaskResult(
                task_name="retrain",
                status=TaskStatus.FAILED,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_seconds=round(duration, 2),
                error=str(exc),
            )
            logger.exception("Scheduled retrain failed.")

        self._record_result(task_result)
        return task_result

    def _task_warm_cache(self) -> TaskResult:
        """Warm prediction cache for top tickers."""
        start = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            from prediction.config import config
            from prediction.inference import PredictionService
            from prediction.utils.cache import CacheClient

            service = PredictionService()
            cache = CacheClient(redis_url=config.redis.url)

            tickers = list(config.tracked_tickers[:self._top_n])

            def prediction_fn(ticker: str) -> dict:
                results = service.predict(ticker, horizon_days=5)
                return service._serialize_predictions(results)

            warmed = cache.warm_cache(tickers, prediction_fn)
            duration = time.monotonic() - start

            task_result = TaskResult(
                task_name="warm_cache",
                status=TaskStatus.COMPLETED,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_seconds=round(duration, 2),
                details={"tickers_warmed": warmed, "total_tickers": len(tickers)},
            )
        except Exception as exc:
            duration = time.monotonic() - start
            task_result = TaskResult(
                task_name="warm_cache",
                status=TaskStatus.FAILED,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_seconds=round(duration, 2),
                error=str(exc),
            )
            logger.exception("Cache warming failed.")

        self._record_result(task_result)
        return task_result

    def _task_refresh_predictions(self) -> TaskResult:
        """Refresh predictions for all tracked tickers and broadcast."""
        start = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            from prediction.config import config
            from prediction.inference import PredictionService

            service = PredictionService()
            tickers = list(config.tracked_tickers[:self._top_n])
            refreshed = 0

            for ticker in tickers:
                try:
                    # Invalidate old cache
                    service._cache.invalidate_predictions(ticker)

                    # Generate fresh predictions
                    results = service.predict(ticker, horizon_days=5)

                    # Broadcast via WebSocket if stream manager attached
                    if self._stream is not None and results:
                        pred_data = {
                            "predictions": [
                                {
                                    "target_date": str(r.target_date),
                                    "predicted_close": str(r.predicted_close),
                                    "confidence_lower": str(r.confidence_lower),
                                    "confidence_upper": str(r.confidence_upper),
                                    "model": r.model_name,
                                }
                                for r in results
                            ]
                        }
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                task = asyncio.ensure_future(
                                    self._stream.broadcast_prediction(ticker, pred_data)
                                )
                                self._background_tasks.add(task)
                                task.add_done_callback(self._background_tasks.discard)
                        except RuntimeError:
                            pass

                    refreshed += 1
                except Exception:
                    logger.warning("Failed to refresh predictions for %s", ticker)

            duration = time.monotonic() - start

            task_result = TaskResult(
                task_name="refresh_predictions",
                status=TaskStatus.COMPLETED,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_seconds=round(duration, 2),
                details={"refreshed": refreshed, "total": len(tickers)},
            )
        except Exception as exc:
            duration = time.monotonic() - start
            task_result = TaskResult(
                task_name="refresh_predictions",
                status=TaskStatus.FAILED,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_seconds=round(duration, 2),
                error=str(exc),
            )
            logger.exception("Prediction refresh failed.")

        self._record_result(task_result)
        return task_result

    def _task_drift_check(self) -> TaskResult:
        """Check for model performance drift."""
        start = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            from prediction.utils.metrics import ModelMonitor

            monitor = ModelMonitor()
            needs_retrain = monitor.check_retrain_needed()

            duration = time.monotonic() - start

            if needs_retrain:
                logger.warning("Drift detected — triggering automatic retrain.")
                # Chain a retrain task
                retrain_result = self._task_retrain_models()
                task_result = TaskResult(
                    task_name="drift_check",
                    status=TaskStatus.COMPLETED,
                    started_at=started_at,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    duration_seconds=round(duration, 2),
                    details={
                        "drift_detected": True,
                        "retrain_triggered": True,
                        "retrain_status": retrain_result.status.value,
                    },
                )
            else:
                task_result = TaskResult(
                    task_name="drift_check",
                    status=TaskStatus.COMPLETED,
                    started_at=started_at,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    duration_seconds=round(duration, 2),
                    details={"drift_detected": False, "retrain_triggered": False},
                )
        except Exception as exc:
            duration = time.monotonic() - start
            task_result = TaskResult(
                task_name="drift_check",
                status=TaskStatus.FAILED,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_seconds=round(duration, 2),
                error=str(exc),
            )

        self._record_result(task_result)
        return task_result

    def _task_full_pipeline(self) -> TaskResult:
        """Run the full pipeline: ETL → Retrain → Warm Cache → Refresh."""
        start = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        sub_results: list[TaskResult] = []

        try:
            # 1. ETL
            r = self._task_incremental_etl()
            sub_results.append(r)
            if r.status == TaskStatus.FAILED:
                raise RuntimeError(f"ETL failed: {r.error}")

            # 2. Retrain
            r = self._task_retrain_models()
            sub_results.append(r)
            if r.status == TaskStatus.FAILED:
                raise RuntimeError(f"Retrain failed: {r.error}")

            # 3. Warm cache
            r = self._task_warm_cache()
            sub_results.append(r)

            # 4. Refresh predictions
            r = self._task_refresh_predictions()
            sub_results.append(r)

            duration = time.monotonic() - start
            task_result = TaskResult(
                task_name="full_pipeline",
                status=TaskStatus.COMPLETED,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_seconds=round(duration, 2),
                details={
                    "steps": [
                        {"task": s.task_name, "status": s.status.value}
                        for s in sub_results
                    ]
                },
            )
        except Exception as exc:
            duration = time.monotonic() - start
            task_result = TaskResult(
                task_name="full_pipeline",
                status=TaskStatus.FAILED,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_seconds=round(duration, 2),
                error=str(exc),
                details={
                    "steps": [
                        {"task": s.task_name, "status": s.status.value}
                        for s in sub_results
                    ]
                },
            )

        self._record_result(task_result)
        return task_result

    # ------------------------------------------------------------------
    # Status & introspection
    # ------------------------------------------------------------------

    def get_scheduled_jobs(self) -> list[dict]:
        """Return info about all scheduled jobs."""
        if self._scheduler is not None and HAS_APSCHEDULER:
            return [
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": str(job.next_run_time),
                    "trigger": str(job.trigger),
                }
                for job in self._scheduler.get_jobs()
            ]
        return [{"note": "Using fallback threading scheduler (no job introspection)"}]

    def get_status(self) -> dict:
        """Return the scheduler status summary."""
        recent = self._task_history[-10:] if self._task_history else []
        return {
            "running": self._running,
            "backend": "apscheduler" if HAS_APSCHEDULER else "threading",
            "jobs": self.get_scheduled_jobs(),
            "recent_tasks": [
                {
                    "task": r.task_name,
                    "status": r.status.value,
                    "duration": r.duration_seconds,
                    "started_at": r.started_at,
                }
                for r in recent
            ],
        }
