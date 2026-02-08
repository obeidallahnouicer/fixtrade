"""
WebSocket & SSE prediction stream manager.

Manages connected WebSocket clients and broadcasts live prediction
updates whenever new data is processed or models produce fresh results.

Architecture:
    FastAPI WebSocket endpoint  ──▶  PredictionStreamManager
                                          │
                                    ┌─────┴──────┐
                                    │ Connected   │
                                    │ clients set │
                                    └─────┬──────┘
                                          │ broadcast()
                                          ▼
                                    JSON message to all

Also exposes an SSE (Server-Sent Events) generator for clients
that can't use WebSocket (e.g. curl, monitoring dashboards).
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, AsyncGenerator

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """A single event pushed to connected clients."""

    event_type: str          # "prediction", "volume", "liquidity", "retrain", "etl"
    symbol: str | None
    data: dict
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps({
            "event": self.event_type,
            "symbol": self.symbol,
            "data": self.data,
            "timestamp": self.timestamp,
        }, default=str)

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        return f"event: {self.event_type}\ndata: {self.to_json()}\n\n"


class PredictionStreamManager:
    """Manages real-time prediction streaming to WebSocket & SSE clients.

    Thread-safe: uses asyncio.Queue per client for backpressure handling.
    Supports symbol-level subscriptions — clients can subscribe to
    specific tickers or receive all events.

    Usage in FastAPI:
        manager = PredictionStreamManager()

        @app.websocket("/ws/predictions")
        async def ws_endpoint(ws: WebSocket):
            await manager.connect(ws)
            try:
                while True:
                    msg = await ws.receive_text()
                    await manager.handle_client_message(ws, msg)
            except WebSocketDisconnect:
                manager.disconnect(ws)

        # From scheduler / watcher:
        await manager.broadcast_prediction("BIAT", prediction_data)
    """

    def __init__(self, max_queue_size: int = 100) -> None:
        self._clients: dict[Any, asyncio.Queue] = {}
        self._subscriptions: dict[Any, set[str]] = defaultdict(set)
        self._max_queue_size = max_queue_size
        self._event_history: list[StreamEvent] = []
        self._max_history = 500
        self._stats = {
            "total_connections": 0,
            "total_events_broadcast": 0,
            "total_messages_sent": 0,
        }

    @property
    def active_connections(self) -> int:
        return len(self._clients)

    @property
    def stats(self) -> dict:
        return {**self._stats, "active_connections": self.active_connections}

    # ------------------------------------------------------------------
    # WebSocket lifecycle
    # ------------------------------------------------------------------

    async def connect(self, websocket: Any) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self._clients[websocket] = asyncio.Queue(maxsize=self._max_queue_size)
        self._subscriptions[websocket] = set()  # empty = all symbols
        self._stats["total_connections"] += 1
        logger.info(
            "WebSocket client connected. Active: %d", self.active_connections
        )

        # Send connection confirmation
        welcome = StreamEvent(
            event_type="connected",
            symbol=None,
            data={
                "message": "Connected to FixTrade real-time prediction stream",
                "active_clients": self.active_connections,
            },
        )
        await websocket.send_text(welcome.to_json())

    def disconnect(self, websocket: Any) -> None:
        """Remove a disconnected WebSocket client."""
        self._clients.pop(websocket, None)
        self._subscriptions.pop(websocket, None)
        logger.info(
            "WebSocket client disconnected. Active: %d", self.active_connections
        )

    async def handle_client_message(self, websocket: Any, raw: str) -> None:
        """Process a message from a WebSocket client.

        Supported commands:
            {"action": "subscribe", "symbols": ["BIAT", "SFBT"]}
            {"action": "unsubscribe", "symbols": ["SFBT"]}
            {"action": "subscribe_all"}
            {"action": "ping"}
        """
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
            return

        action = msg.get("action", "")

        if action == "subscribe":
            symbols = {s.upper() for s in msg.get("symbols", [])}
            self._subscriptions[websocket] |= symbols
            await websocket.send_text(json.dumps({
                "event": "subscribed",
                "symbols": sorted(self._subscriptions[websocket]),
            }))

        elif action == "unsubscribe":
            symbols = {s.upper() for s in msg.get("symbols", [])}
            self._subscriptions[websocket] -= symbols
            await websocket.send_text(json.dumps({
                "event": "unsubscribed",
                "symbols": sorted(self._subscriptions[websocket]),
            }))

        elif action == "subscribe_all":
            self._subscriptions[websocket] = set()
            await websocket.send_text(json.dumps({
                "event": "subscribed_all",
            }))

        elif action == "ping":
            await websocket.send_text(json.dumps({
                "event": "pong",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }))

        else:
            await websocket.send_text(json.dumps({
                "error": f"Unknown action: {action}",
                "supported": ["subscribe", "unsubscribe", "subscribe_all", "ping"],
            }))

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    async def broadcast(self, event: StreamEvent) -> int:
        """Send an event to all matching clients.

        Returns the number of clients that received the message.
        """
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        self._stats["total_events_broadcast"] += 1
        sent = 0
        dead: list[Any] = []

        for ws, queue in self._clients.items():
            # Check subscription filter
            subs = self._subscriptions.get(ws, set())
            if subs and event.symbol and event.symbol.upper() not in subs:
                continue

            try:
                await ws.send_text(event.to_json())
                sent += 1
                self._stats["total_messages_sent"] += 1
            except Exception:
                dead.append(ws)

        # Clean up dead connections
        for ws in dead:
            self.disconnect(ws)

        return sent

    async def broadcast_prediction(self, symbol: str, prediction_data: dict) -> int:
        """Broadcast a price prediction update."""
        event = StreamEvent(
            event_type="prediction",
            symbol=symbol,
            data=prediction_data,
        )
        return await self.broadcast(event)

    async def broadcast_volume(self, symbol: str, volume_data: dict) -> int:
        """Broadcast a volume prediction update."""
        event = StreamEvent(
            event_type="volume",
            symbol=symbol,
            data=volume_data,
        )
        return await self.broadcast(event)

    async def broadcast_liquidity(self, symbol: str, liquidity_data: dict) -> int:
        """Broadcast a liquidity probability update."""
        event = StreamEvent(
            event_type="liquidity",
            symbol=symbol,
            data=liquidity_data,
        )
        return await self.broadcast(event)

    async def broadcast_system_event(
        self, event_type: str, data: dict
    ) -> int:
        """Broadcast a system event (retrain, etl, error)."""
        event = StreamEvent(
            event_type=event_type,
            symbol=None,
            data=data,
        )
        return await self.broadcast(event)

    # ------------------------------------------------------------------
    # SSE Generator
    # ------------------------------------------------------------------

    async def sse_generator(
        self, symbols: set[str] | None = None
    ) -> AsyncGenerator[str, None]:
        """Async generator that yields Server-Sent Events.

        Use with FastAPI StreamingResponse:

            @app.get("/stream/predictions")
            async def stream(request: Request):
                return StreamingResponse(
                    manager.sse_generator(symbols={"BIAT"}),
                    media_type="text/event-stream",
                )
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue_size)

        # Register a virtual SSE client
        sse_id = f"sse-{id(queue)}-{time.monotonic_ns()}"
        self._clients[sse_id] = queue
        self._subscriptions[sse_id] = symbols or set()
        self._stats["total_connections"] += 1

        try:
            # Send initial keepalive
            yield ": connected\n\n"

            while True:
                # Poll event history for new events matching our subscription
                await asyncio.sleep(1)

                # Check for new events (simple polling approach)
                for event in self._event_history[-10:]:
                    if symbols and event.symbol and event.symbol.upper() not in symbols:
                        continue
                    yield event.to_sse()

        finally:
            self._clients.pop(sse_id, None)
            self._subscriptions.pop(sse_id, None)

    # ------------------------------------------------------------------
    # History & replay
    # ------------------------------------------------------------------

    def get_recent_events(
        self, limit: int = 50, symbol: str | None = None
    ) -> list[dict]:
        """Return recent events, optionally filtered by symbol."""
        events = self._event_history
        if symbol:
            events = [e for e in events if e.symbol and e.symbol.upper() == symbol.upper()]
        return [json.loads(e.to_json()) for e in events[-limit:]]
