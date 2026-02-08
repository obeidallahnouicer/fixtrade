"""
Tests for the real-time prediction pipeline.

Covers:
- PredictionStreamManager (WebSocket broadcast, subscriptions, SSE)
- RealtimeScheduler (task execution, status, history)
- DataWatcher (snapshot, diff, change detection)
- ModelMonitor.check_retrain_needed
- StreamEvent serialization
"""

import asyncio
import hashlib
import json
import os
import tempfile
import time
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# =====================================================================
# StreamEvent
# =====================================================================

class TestStreamEvent:
    """Tests for the StreamEvent data class."""

    def test_to_json_produces_valid_json(self):
        from prediction.realtime.stream import StreamEvent

        event = StreamEvent(
            event_type="prediction",
            symbol="BIAT",
            data={"predicted_close": 97.85},
        )
        raw = event.to_json()
        parsed = json.loads(raw)

        assert parsed["event"] == "prediction"
        assert parsed["symbol"] == "BIAT"
        assert parsed["data"]["predicted_close"] == 97.85
        assert "timestamp" in parsed

    def test_to_sse_format(self):
        from prediction.realtime.stream import StreamEvent

        event = StreamEvent(
            event_type="volume",
            symbol="SFBT",
            data={"predicted_volume": 12345},
        )
        sse = event.to_sse()

        assert sse.startswith("event: volume\n")
        assert "data: " in sse
        assert sse.endswith("\n\n")

    def test_default_timestamp_is_utc(self):
        from prediction.realtime.stream import StreamEvent

        event = StreamEvent(event_type="test", symbol=None, data={})
        # Should be a valid ISO timestamp
        dt = datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
        assert dt is not None


# =====================================================================
# PredictionStreamManager
# =====================================================================

class TestPredictionStreamManager:
    """Tests for the WebSocket/SSE stream manager."""

    def test_initial_state(self):
        from prediction.realtime.stream import PredictionStreamManager

        mgr = PredictionStreamManager()
        assert mgr.active_connections == 0
        assert mgr.stats["total_events_broadcast"] == 0

    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self):
        from prediction.realtime.stream import PredictionStreamManager

        mgr = PredictionStreamManager()
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_text = AsyncMock()

        await mgr.connect(ws)
        assert mgr.active_connections == 1
        ws.accept.assert_awaited_once()
        ws.send_text.assert_awaited_once()  # welcome message

        mgr.disconnect(ws)
        assert mgr.active_connections == 0

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all_clients(self):
        from prediction.realtime.stream import PredictionStreamManager, StreamEvent

        mgr = PredictionStreamManager()

        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_text = AsyncMock()
        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_text = AsyncMock()

        await mgr.connect(ws1)
        await mgr.connect(ws2)

        sent = await mgr.broadcast_prediction("BIAT", {"close": 97.0})
        assert sent == 2

    @pytest.mark.asyncio
    async def test_subscription_filter(self):
        from prediction.realtime.stream import PredictionStreamManager, StreamEvent

        mgr = PredictionStreamManager()

        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_text = AsyncMock()

        await mgr.connect(ws)

        # Subscribe to BIAT only
        await mgr.handle_client_message(
            ws, json.dumps({"action": "subscribe", "symbols": ["BIAT"]})
        )

        # Broadcast SFBT event — should NOT reach this client
        sent = await mgr.broadcast_prediction("SFBT", {"close": 18.50})
        assert sent == 0

        # Broadcast BIAT event — SHOULD reach this client
        sent = await mgr.broadcast_prediction("BIAT", {"close": 97.0})
        assert sent == 1

    @pytest.mark.asyncio
    async def test_handle_ping(self):
        from prediction.realtime.stream import PredictionStreamManager

        mgr = PredictionStreamManager()
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_text = AsyncMock()

        await mgr.connect(ws)
        await mgr.handle_client_message(ws, json.dumps({"action": "ping"}))

        # Should have received welcome + pong
        assert ws.send_text.await_count == 2
        last_call = ws.send_text.call_args_list[-1]
        msg = json.loads(last_call.args[0])
        assert msg["event"] == "pong"

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self):
        from prediction.realtime.stream import PredictionStreamManager

        mgr = PredictionStreamManager()
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_text = AsyncMock()

        await mgr.connect(ws)
        await mgr.handle_client_message(ws, "not json at all")

        last_call = ws.send_text.call_args_list[-1]
        msg = json.loads(last_call.args[0])
        assert "error" in msg

    def test_get_recent_events(self):
        from prediction.realtime.stream import PredictionStreamManager, StreamEvent

        mgr = PredictionStreamManager()
        # Manually add events to history
        for i in range(5):
            mgr._event_history.append(StreamEvent(
                event_type="prediction",
                symbol=f"TICK{i}",
                data={"i": i},
            ))

        recent = mgr.get_recent_events(limit=3)
        assert len(recent) == 3
        assert recent[-1]["symbol"] == "TICK4"

    def test_get_recent_events_filtered_by_symbol(self):
        from prediction.realtime.stream import PredictionStreamManager, StreamEvent

        mgr = PredictionStreamManager()
        mgr._event_history.append(StreamEvent("prediction", "BIAT", {"x": 1}))
        mgr._event_history.append(StreamEvent("prediction", "SFBT", {"x": 2}))
        mgr._event_history.append(StreamEvent("prediction", "BIAT", {"x": 3}))

        biat_events = mgr.get_recent_events(limit=10, symbol="BIAT")
        assert len(biat_events) == 2
        assert all(e["symbol"] == "BIAT" for e in biat_events)


# =====================================================================
# RealtimeScheduler
# =====================================================================

class TestRealtimeScheduler:
    """Tests for the scheduler task orchestrator."""

    def test_initial_state(self):
        from prediction.realtime.scheduler import RealtimeScheduler

        scheduler = RealtimeScheduler()
        assert not scheduler.is_running
        assert scheduler.task_history == []

    def test_run_unknown_task(self):
        from prediction.realtime.scheduler import RealtimeScheduler, TaskStatus

        scheduler = RealtimeScheduler()
        result = scheduler.run_now("nonexistent_task")
        assert result.status == TaskStatus.FAILED
        assert "Unknown task" in result.error

    def test_get_status_when_stopped(self):
        from prediction.realtime.scheduler import RealtimeScheduler

        scheduler = RealtimeScheduler()
        status = scheduler.get_status()
        assert status["running"] is False

    def test_task_result_records_in_history(self):
        from prediction.realtime.scheduler import (
            RealtimeScheduler, TaskResult, TaskStatus,
        )

        scheduler = RealtimeScheduler()
        result = TaskResult(
            task_name="test_task",
            status=TaskStatus.COMPLETED,
            started_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=1.5,
        )
        scheduler._record_result(result)
        assert len(scheduler.task_history) == 1
        assert scheduler.task_history[0].task_name == "test_task"

    @patch("prediction.realtime.scheduler.RealtimeScheduler._task_warm_cache")
    def test_run_now_warm_cache(self, mock_warm):
        from prediction.realtime.scheduler import (
            RealtimeScheduler, TaskResult, TaskStatus,
        )

        mock_warm.return_value = TaskResult(
            task_name="warm_cache",
            status=TaskStatus.COMPLETED,
            started_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=2.0,
            details={"tickers_warmed": 5},
        )

        scheduler = RealtimeScheduler()
        result = scheduler.run_now("warm_cache")
        assert result.status == TaskStatus.COMPLETED
        mock_warm.assert_called_once()


# =====================================================================
# DataWatcher
# =====================================================================

class TestDataWatcher:
    """Tests for the file system watcher."""

    def test_initial_state(self):
        from prediction.realtime.watcher import DataWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = DataWatcher(watch_dir=tmpdir, poll_interval=1.0)
            assert not watcher.is_running
            assert watcher.change_log == []

    def test_snapshot_detects_csv_files(self):
        from prediction.realtime.watcher import DataWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some CSV files
            (Path(tmpdir) / "stock_data.csv").write_text("a,b\n1,2\n")
            (Path(tmpdir) / "other.txt").write_text("not a csv")

            watcher = DataWatcher(watch_dir=tmpdir)
            snapshot = watcher._snapshot()

            # Only the CSV should be tracked
            assert len(snapshot) == 1
            assert any("stock_data.csv" in k for k in snapshot)

    def test_diff_detects_new_file(self):
        from prediction.realtime.watcher import DataWatcher, FileChange

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "new_data.csv"

            watcher = DataWatcher(watch_dir=tmpdir)

            old_snapshot = {}
            csv_path.write_text("col1,col2\n1,2\n")
            new_snapshot = watcher._snapshot()

            changes = DataWatcher._diff(old_snapshot, new_snapshot)
            assert len(changes) == 1
            assert changes[0].change_type == "created"

    def test_diff_detects_modified_file(self):
        from prediction.realtime.watcher import DataWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text("col1\n1\n")

            watcher = DataWatcher(watch_dir=tmpdir)
            old_snapshot = watcher._snapshot()

            # Modify the file
            csv_path.write_text("col1\n1\n2\n3\n")
            new_snapshot = watcher._snapshot()

            changes = DataWatcher._diff(old_snapshot, new_snapshot)
            assert len(changes) == 1
            assert changes[0].change_type == "modified"

    def test_diff_detects_deleted_file(self):
        from prediction.realtime.watcher import DataWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text("col1\n1\n")

            watcher = DataWatcher(watch_dir=tmpdir)
            old_snapshot = watcher._snapshot()

            csv_path.unlink()
            new_snapshot = watcher._snapshot()

            changes = DataWatcher._diff(old_snapshot, new_snapshot)
            assert len(changes) == 1
            assert changes[0].change_type == "deleted"

    def test_ignores_tmp_files(self):
        from prediction.realtime.watcher import DataWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "data.tmp").write_text("temp content")
            (Path(tmpdir) / "data.part").write_text("partial")
            (Path(tmpdir) / "data.csv").write_text("real data")

            watcher = DataWatcher(watch_dir=tmpdir)
            snapshot = watcher._snapshot()

            assert len(snapshot) == 1
            assert any("data.csv" in k for k in snapshot)

    def test_get_status(self):
        from prediction.realtime.watcher import DataWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = DataWatcher(watch_dir=tmpdir, poll_interval=5.0)
            status = watcher.get_status()

            assert status["running"] is False
            assert status["poll_interval_seconds"] == 5.0
            assert str(tmpdir) in status["watch_dir"]

    def test_file_hash_deterministic(self):
        from prediction.realtime.watcher import DataWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("hello,world\n1,2\n")

            h1 = DataWatcher._file_hash(csv_path)
            h2 = DataWatcher._file_hash(csv_path)
            assert h1 == h2
            assert len(h1) == 32  # MD5 hex digest


# =====================================================================
# ModelMonitor.check_retrain_needed
# =====================================================================

class TestModelMonitorRetrainCheck:
    """Tests for the check_retrain_needed method."""

    def test_no_history_returns_false(self):
        from prediction.utils.metrics import ModelMonitor

        monitor = ModelMonitor()
        assert monitor.check_retrain_needed() is False

    def test_healthy_model_returns_false(self):
        from prediction.utils.metrics import ModelMonitor

        monitor = ModelMonitor(rmse_threshold=2.0, dir_acc_threshold=0.50)
        y_true = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        y_pred = np.array([100.1, 101.1, 102.1, 103.1, 104.1])

        monitor.evaluate_model("test_model", y_true, y_pred)
        assert monitor.check_retrain_needed() is False

    def test_drifted_model_triggers_retrain(self):
        from prediction.utils.metrics import ModelMonitor

        monitor = ModelMonitor(rmse_threshold=0.001, dir_acc_threshold=0.99)
        y_true = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        y_pred = np.array([50.0, 51.0, 52.0, 53.0, 54.0])  # way off

        monitor.evaluate_model("bad_model", y_true, y_pred)
        assert monitor.check_retrain_needed() is True


# =====================================================================
# TaskResult & TaskStatus
# =====================================================================

class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_task_result_fields(self):
        from prediction.realtime.scheduler import TaskResult, TaskStatus

        result = TaskResult(
            task_name="etl",
            status=TaskStatus.COMPLETED,
            started_at="2025-01-01T00:00:00",
            finished_at="2025-01-01T00:01:00",
            duration_seconds=60.0,
            details={"rows": 1000},
        )
        assert result.task_name == "etl"
        assert result.status == TaskStatus.COMPLETED
        assert result.duration_seconds == 60.0
        assert result.details["rows"] == 1000
        assert result.error is None

    def test_task_status_values(self):
        from prediction.realtime.scheduler import TaskStatus

        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
