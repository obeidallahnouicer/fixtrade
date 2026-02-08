"""
Filesystem watcher for automatic data ingestion.

Monitors the ``data/raw/`` directory for new or modified CSV files.
When a change is detected:
  1. Triggers incremental ETL
  2. Optionally retrains models
  3. Broadcasts an event to connected WebSocket clients

Uses a simple polling strategy (works on all OSes including Windows
where inotify is unavailable). Poll interval is configurable.

Usage:
    watcher = DataWatcher(watch_dir="data/raw")
    watcher.start()   # background thread
    watcher.stop()
"""

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FileChange:
    """A detected file system change."""

    path: str
    change_type: str  # "created", "modified", "deleted"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    size_bytes: int = 0


class DataWatcher:
    """Watches ``data/raw/`` for new CSV files and triggers the pipeline.

    Uses MD5 fingerprints to detect true file changes (not just mtime).
    Ignores temporary files (.tmp, .part, ~lock).

    Architecture:
        ┌────────────┐   poll every N sec    ┌───────────┐
        │ data/raw/  │ ──────────────────▶   │ Watcher   │
        │  *.csv     │                       │ (thread)  │
        └────────────┘                       └─────┬─────┘
                                                   │ on_change()
                                             ┌─────▼──────┐
                                             │ ETL → Model │
                                             │  → Broadcast│
                                             └────────────┘
    """

    IGNORED_EXTENSIONS = {".tmp", ".part", ".lock", ".swp"}
    WATCHED_EXTENSIONS = {".csv", ".CSV"}

    def __init__(
        self,
        watch_dir: str | Path | None = None,
        poll_interval: float = 30.0,
        auto_retrain: bool = False,
        scheduler: Any | None = None,
        stream_manager: Any | None = None,
    ) -> None:
        """
        Args:
            watch_dir: Directory to watch. Defaults to ``data/raw``.
            poll_interval: Seconds between directory scans.
            auto_retrain: Whether to trigger model retraining on new data.
            scheduler: Optional RealtimeScheduler for triggering tasks.
            stream_manager: Optional PredictionStreamManager for broadcasts.
        """
        if watch_dir is None:
            from prediction.config import config
            watch_dir = config.paths.raw
        self._watch_dir = Path(watch_dir)
        self._poll_interval = poll_interval
        self._auto_retrain = auto_retrain
        self._scheduler = scheduler
        self._stream = stream_manager

        self._running = False
        self._thread: threading.Thread | None = None
        self._file_hashes: dict[str, str] = {}
        self._change_log: list[FileChange] = []
        self._max_log = 500
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def change_log(self) -> list[FileChange]:
        return list(self._change_log)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start watching in a background thread."""
        if self._running:
            logger.warning("DataWatcher already running.")
            return

        if not self._watch_dir.exists():
            logger.warning("Watch dir %s does not exist. Creating it.", self._watch_dir)
            self._watch_dir.mkdir(parents=True, exist_ok=True)

        # Take initial snapshot
        self._file_hashes = self._snapshot()
        self._running = True

        self._thread = threading.Thread(
            target=self._poll_loop, name="data-watcher", daemon=True
        )
        self._thread.start()
        logger.info(
            "DataWatcher started — monitoring %s (every %.0fs)",
            self._watch_dir, self._poll_interval,
        )

    def stop(self) -> None:
        """Stop the watcher."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval + 5)
            self._thread = None
        logger.info("DataWatcher stopped.")

    # ------------------------------------------------------------------
    # Polling loop
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Main loop: snapshot → diff → handle changes → sleep."""
        while self._running:
            try:
                new_snapshot = self._snapshot()
                changes = self._diff(self._file_hashes, new_snapshot)

                if changes:
                    logger.info(
                        "Detected %d file change(s) in %s",
                        len(changes), self._watch_dir,
                    )
                    for change in changes:
                        self._record_change(change)
                    self._on_changes(changes)

                self._file_hashes = new_snapshot

            except Exception:
                logger.exception("Error in DataWatcher poll loop.")

            time.sleep(self._poll_interval)

    # ------------------------------------------------------------------
    # Snapshot & diffing
    # ------------------------------------------------------------------

    def _snapshot(self) -> dict[str, str]:
        """Build a {filepath: md5_hash} map of all watched files."""
        result: dict[str, str] = {}
        if not self._watch_dir.exists():
            return result

        for fpath in self._watch_dir.iterdir():
            if not fpath.is_file():
                continue
            ext = fpath.suffix.lower()
            if ext in self.IGNORED_EXTENSIONS:
                continue
            if ext not in {e.lower() for e in self.WATCHED_EXTENSIONS}:
                continue

            try:
                h = self._file_hash(fpath)
                result[str(fpath)] = h
            except OSError:
                pass

        return result

    @staticmethod
    def _file_hash(path: Path, chunk_size: int = 8192) -> str:
        """Compute MD5 hash of a file (fast, not cryptographic)."""
        md5 = hashlib.md5()
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()

    @staticmethod
    def _diff(
        old: dict[str, str], new: dict[str, str]
    ) -> list[FileChange]:
        """Compare two snapshots and return changes."""
        changes: list[FileChange] = []

        # New or modified files
        for path, hash_val in new.items():
            if path not in old:
                size = os.path.getsize(path) if os.path.exists(path) else 0
                changes.append(FileChange(
                    path=path, change_type="created", size_bytes=size,
                ))
            elif old[path] != hash_val:
                size = os.path.getsize(path) if os.path.exists(path) else 0
                changes.append(FileChange(
                    path=path, change_type="modified", size_bytes=size,
                ))

        # Deleted files
        for path in old:
            if path not in new:
                changes.append(FileChange(
                    path=path, change_type="deleted",
                ))

        return changes

    def _record_change(self, change: FileChange) -> None:
        with self._lock:
            self._change_log.append(change)
            if len(self._change_log) > self._max_log:
                self._change_log = self._change_log[-self._max_log:]

    # ------------------------------------------------------------------
    # Change handlers
    # ------------------------------------------------------------------

    def _on_changes(self, changes: list[FileChange]) -> None:
        """Handle detected file changes."""
        created_or_modified = [
            c for c in changes if c.change_type in ("created", "modified")
        ]
        if not created_or_modified:
            return

        logger.info(
            "New/modified CSV files: %s",
            [os.path.basename(c.path) for c in created_or_modified],
        )

        # 1. Run incremental ETL
        if self._scheduler is not None:
            logger.info("Triggering incremental ETL via scheduler...")
            result = self._scheduler.run_now("etl")
            logger.info(
                "ETL %s (%.1fs)", result.status.value, result.duration_seconds
            )

            # 2. Optionally retrain
            if self._auto_retrain:
                logger.info("Auto-retrain enabled — triggering model retraining...")
                retrain = self._scheduler.run_now("retrain")
                logger.info(
                    "Retrain %s (%.1fs)",
                    retrain.status.value, retrain.duration_seconds,
                )

            # 3. Refresh predictions
            logger.info("Refreshing predictions...")
            self._scheduler.run_now("refresh_predictions")
        else:
            # No scheduler — run ETL directly
            try:
                from prediction.pipeline import ETLPipeline

                pipeline = ETLPipeline()
                result = pipeline.run_incremental()
                logger.info(
                    "ETL complete: %d rows.",
                    len(result) if result is not None else 0,
                )
            except Exception:
                logger.exception("ETL triggered by watcher failed.")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return watcher status."""
        recent = self._change_log[-10:] if self._change_log else []
        return {
            "running": self._running,
            "watch_dir": str(self._watch_dir),
            "poll_interval_seconds": self._poll_interval,
            "auto_retrain": self._auto_retrain,
            "tracked_files": len(self._file_hashes),
            "total_changes_detected": len(self._change_log),
            "recent_changes": [
                {
                    "path": os.path.basename(c.path),
                    "type": c.change_type,
                    "timestamp": c.timestamp,
                    "size_bytes": c.size_bytes,
                }
                for c in recent
            ],
        }
