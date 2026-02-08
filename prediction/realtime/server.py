"""
Standalone lightweight FastAPI server for real-time prediction streaming.

This module creates a minimal FastAPI app that includes ONLY the
realtime endpoints (WebSocket, SSE, scheduler control, watcher status).
It does NOT depend on ``app.core.config``, ``slowapi``, or any other
heavy application dependency — making it easy to start with:

    python -m prediction stream --port 8000

Architecture:
    - Creates its own FastAPI instance with lifespan
    - Starts RealtimeScheduler + DataWatcher + PredictionStreamManager
    - Mounts the WebSocket, SSE, and control endpoints
"""

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from prediction.realtime.scheduler import RealtimeScheduler
from prediction.realtime.stream import PredictionStreamManager
from prediction.realtime.watcher import DataWatcher

logger = logging.getLogger(__name__)

# Module-level singletons (initialized during lifespan)
_stream_manager: PredictionStreamManager | None = None
_scheduler: RealtimeScheduler | None = None
_watcher: DataWatcher | None = None
_top_n: int = 10


def create_standalone_app(top_n: int = 10) -> FastAPI:
    """Create a lightweight FastAPI app for real-time streaming only.

    Args:
        top_n: Number of top tickers the scheduler manages.

    Returns:
        A FastAPI app with WebSocket, SSE, and control endpoints.
    """
    global _top_n
    _top_n = top_n

    app = FastAPI(
        title="FixTrade Real-Time Prediction Stream",
        version="1.0.0",
        description="WebSocket & SSE server for live BVMT prediction updates",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=_lifespan,
    )

    # Register routes
    app.add_api_websocket_route("/ws/predictions", _ws_predictions)
    app.add_api_route(
        "/stream/predictions",
        _sse_predictions,
        methods=["GET"],
        summary="SSE prediction stream",
    )
    app.add_api_route(
        "/scheduler/status",
        _scheduler_status,
        methods=["GET"],
        summary="Scheduler status",
    )
    app.add_api_route(
        "/scheduler/run/{task_name}",
        _scheduler_run_task,
        methods=["POST"],
        summary="Trigger a scheduler task",
    )
    app.add_api_route(
        "/watcher/status",
        _watcher_status,
        methods=["GET"],
        summary="Data watcher status",
    )
    app.add_api_route(
        "/stream/status",
        _stream_status,
        methods=["GET"],
        summary="Stream connection stats",
    )
    app.add_api_route(
        "/health",
        _health,
        methods=["GET"],
        summary="Health check",
    )

    return app


# ------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Start and stop the realtime pipeline components."""
    global _stream_manager, _scheduler, _watcher

    _stream_manager = PredictionStreamManager()
    _scheduler = RealtimeScheduler(
        stream_manager=_stream_manager,
        top_n_tickers=_top_n,
    )
    _watcher = DataWatcher(
        poll_interval=30.0,
        auto_retrain=False,
        scheduler=_scheduler,
        stream_manager=_stream_manager,
    )

    _scheduler.start()
    _watcher.start()
    logger.info("Realtime components started (scheduler + watcher + stream).")

    yield

    _watcher.stop()
    _scheduler.stop()
    logger.info("Realtime components stopped.")


# ------------------------------------------------------------------
# Route handlers
# ------------------------------------------------------------------


async def _ws_predictions(websocket: WebSocket) -> None:
    """WebSocket endpoint for live prediction streaming."""
    if _stream_manager is None:
        await websocket.close(code=1013, reason="Server not ready")
        return

    await _stream_manager.connect(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            await _stream_manager.handle_client_message(websocket, raw)
    except WebSocketDisconnect:
        _stream_manager.disconnect(websocket)
    except Exception:
        _stream_manager.disconnect(websocket)


async def _sse_predictions(
    symbols: Annotated[
        str | None,
        Query(description="Comma-separated ticker symbols to subscribe to"),
    ] = None,
) -> StreamingResponse:
    """SSE endpoint for HTTP-only clients."""
    if _stream_manager is None:
        return StreamingResponse(
            iter(["data: server not ready\n\n"]),
            media_type="text/event-stream",
        )

    filter_symbols: set[str] | None = None
    if symbols:
        filter_symbols = {s.strip().upper() for s in symbols.split(",")}

    return StreamingResponse(
        _stream_manager.sse_generator(symbols=filter_symbols),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _scheduler_status() -> dict:
    """Return scheduler status and recent task history."""
    if _scheduler is None:
        return {"running": False, "note": "Scheduler not initialized."}
    return _scheduler.get_status()


def _scheduler_run_task(task_name: str) -> dict:
    """Trigger a scheduler task on demand."""
    if _scheduler is None:
        return {"error": "Scheduler not initialized."}

    result = _scheduler.run_now(task_name)
    return {
        "task": result.task_name,
        "status": result.status.value,
        "duration_seconds": result.duration_seconds,
        "details": result.details,
        "error": result.error,
    }


def _watcher_status() -> dict:
    """Return the data watcher status."""
    if _watcher is None:
        return {"running": False, "note": "Data watcher not initialized."}
    return _watcher.get_status()


def _stream_status() -> dict:
    """Return streaming stats."""
    if _stream_manager is None:
        return {"active_connections": 0, "note": "Stream not initialized."}
    return {
        **_stream_manager.stats,
        "recent_events": _stream_manager.get_recent_events(limit=20),
    }


def _health() -> dict:
    """Basic health check."""
    return {
        "status": "ok",
        "scheduler_running": _scheduler.is_running if _scheduler else False,
        "watcher_running": _watcher.is_running if _watcher else False,
        "active_ws_connections": _stream_manager.active_connections if _stream_manager else 0,
    }
