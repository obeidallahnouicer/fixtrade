"""
FastAPI router for real-time prediction streaming.

Provides:
- WebSocket endpoint for live prediction updates
- SSE (Server-Sent Events) endpoint for HTTP-only clients
- Scheduler status / control endpoints
- Data watcher status endpoint
"""

import json
import logging
from typing import Annotated

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from prediction.realtime.stream import PredictionStreamManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/realtime", tags=["realtime"])

# ── Singletons (initialized by the app lifespan) ─────────────────
# These are set by app.main during startup.
_stream_manager: PredictionStreamManager | None = None
_scheduler = None
_watcher = None


def set_realtime_components(
    stream_manager: PredictionStreamManager,
    scheduler=None,
    watcher=None,
) -> None:
    """Called by the app lifespan to inject the singleton instances."""
    global _stream_manager, _scheduler, _watcher
    _stream_manager = stream_manager
    _scheduler = scheduler
    _watcher = watcher


def get_stream_manager() -> PredictionStreamManager:
    if _stream_manager is None:
        raise RuntimeError(
            "PredictionStreamManager not initialized. "
            "Ensure the app lifespan starts the realtime components."
        )
    return _stream_manager


# ------------------------------------------------------------------
# WebSocket endpoint
# ------------------------------------------------------------------


@router.websocket("/ws/predictions")
async def ws_predictions(websocket: WebSocket) -> None:
    """WebSocket endpoint for live prediction streaming.

    Clients can subscribe to specific symbols or receive all updates.

    Protocol (JSON):
        → {"action": "subscribe", "symbols": ["BIAT", "SFBT"]}
        ← {"event": "subscribed", "symbols": ["BIAT", "SFBT"]}

        → {"action": "ping"}
        ← {"event": "pong", "timestamp": "..."}

        ← {"event": "prediction", "symbol": "BIAT", "data": {...}}
    """
    manager = get_stream_manager()
    await manager.connect(websocket)

    try:
        while True:
            raw = await websocket.receive_text()
            await manager.handle_client_message(websocket, raw)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


# ------------------------------------------------------------------
# SSE endpoint
# ------------------------------------------------------------------


@router.get(
    "/stream/predictions",
    summary="Server-Sent Events prediction stream",
    description="HTTP streaming endpoint for clients that can't use WebSocket.",
)
async def sse_predictions(
    symbols: Annotated[
        str | None,
        Query(description="Comma-separated ticker symbols to subscribe to"),
    ] = None,
) -> StreamingResponse:
    """SSE endpoint: streams prediction events as text/event-stream."""
    manager = get_stream_manager()

    filter_symbols: set[str] | None = None
    if symbols:
        filter_symbols = {s.strip().upper() for s in symbols.split(",")}

    return StreamingResponse(
        manager.sse_generator(symbols=filter_symbols),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ------------------------------------------------------------------
# Scheduler control endpoints
# ------------------------------------------------------------------


@router.get(
    "/scheduler/status",
    summary="Get scheduler status",
    description="Return the current state of the real-time scheduler.",
)
def scheduler_status() -> dict:
    """Return scheduler status and recent task history."""
    if _scheduler is None:
        return {"running": False, "note": "Scheduler not initialized."}
    return _scheduler.get_status()


@router.post(
    "/scheduler/run/{task_name}",
    summary="Trigger a scheduler task",
    description="Run a named task immediately: etl, retrain, warm_cache, refresh_predictions, full_pipeline.",
)
def scheduler_run_task(task_name: str) -> dict:
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


# ------------------------------------------------------------------
# Watcher status
# ------------------------------------------------------------------


@router.get(
    "/watcher/status",
    summary="Get data watcher status",
    description="Return the file watcher state and recent change log.",
)
def watcher_status() -> dict:
    """Return the data watcher status."""
    if _watcher is None:
        return {"running": False, "note": "Data watcher not initialized."}
    return _watcher.get_status()


# ------------------------------------------------------------------
# Stream status
# ------------------------------------------------------------------


@router.get(
    "/stream/status",
    summary="Get stream status",
    description="Return WebSocket connection stats and recent events.",
)
def stream_status() -> dict:
    """Return streaming stats."""
    manager = get_stream_manager()
    return {
        **manager.stats,
        "recent_events": manager.get_recent_events(limit=20),
    }
