"""
Anomaly alert notification system.

Dispatches anomaly alerts through multiple channels:
    1. WebSocket/SSE real-time broadcast (via PredictionStreamManager)
    2. HTTP webhook POST to configurable URLs
    3. In-process callback hooks (for logging, metrics, etc.)

Architecture:
    AnomalyDetectionAdapter  ──▶  AnomalyNotifier
                                        │
                                  ┌─────┴──────────┐
                                  │ Channels:       │
                                  │  • WebSocket    │
                                  │  • Webhook      │
                                  │  • Callback     │
                                  └─────────────────┘

Usage:
    notifier = AnomalyNotifier(stream_manager=mgr)
    notifier.add_webhook("https://hooks.slack.com/...")
    await notifier.notify(alerts)
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Coroutine
from urllib.parse import urlparse

import httpx

from app.domain.trading.entities import AnomalyAlert

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════

SEVERITY_PRIORITY = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}


@dataclass
class NotificationResult:
    """Result of a notification dispatch."""

    channel: str
    success: bool
    recipients: int = 0
    error: str | None = None
    latency_ms: float = 0.0


@dataclass
class NotificationSummary:
    """Summary of all notification dispatches for a batch of alerts."""

    total_alerts: int = 0
    results: list[NotificationResult] = field(default_factory=list)

    @property
    def total_sent(self) -> int:
        return sum(r.recipients for r in self.results if r.success)

    @property
    def all_success(self) -> bool:
        return all(r.success for r in self.results)


def _severity_label(severity: Decimal) -> str:
    """Map numeric severity to a human label."""
    s = float(severity)
    if s >= 0.8:
        return "critical"
    if s >= 0.6:
        return "high"
    if s >= 0.4:
        return "medium"
    return "low"


def _alert_to_dict(alert: AnomalyAlert) -> dict:
    """Serialize an AnomalyAlert for JSON transport."""
    return {
        "id": str(alert.id),
        "symbol": alert.symbol,
        "detected_at": alert.detected_at.isoformat(),
        "anomaly_type": alert.anomaly_type,
        "severity": float(alert.severity),
        "severity_label": _severity_label(alert.severity),
        "description": alert.description,
    }


# Type alias for async callback hooks
NotificationCallback = Callable[
    [list[AnomalyAlert]],
    Coroutine[Any, Any, None],
]


# ══════════════════════════════════════════════════════════════════════
# Notifier
# ══════════════════════════════════════════════════════════════════════


class AnomalyNotifier:
    """Multi-channel anomaly alert notification dispatcher.

    Args:
        stream_manager: Optional PredictionStreamManager for WebSocket/SSE.
        webhook_urls: Initial list of webhook URLs to POST alerts to.
        min_severity: Minimum severity threshold for notifications (0.0–1.0).
        webhook_timeout: HTTP timeout in seconds for webhook calls.
    """

    def __init__(
        self,
        stream_manager: Any | None = None,
        webhook_urls: list[str] | None = None,
        min_severity: float = 0.0,
        webhook_timeout: float = 10.0,
    ) -> None:
        self._stream_manager = stream_manager
        self._webhook_urls: list[str] = list(webhook_urls or [])
        self._callbacks: list[NotificationCallback] = []
        self._min_severity = min_severity
        self._webhook_timeout = webhook_timeout
        self._stats = {
            "total_notifications": 0,
            "total_alerts_sent": 0,
            "websocket_broadcasts": 0,
            "webhook_calls": 0,
            "callback_invocations": 0,
            "errors": 0,
        }

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def add_webhook(self, url: str) -> None:
        """Register a webhook URL for alert delivery.

        Args:
            url: Full URL (must be http/https).

        Raises:
            ValueError: If the URL is invalid.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            msg = f"Invalid webhook URL scheme: {parsed.scheme}"
            raise ValueError(msg)
        if url not in self._webhook_urls:
            self._webhook_urls.append(url)
            logger.info("Webhook registered: %s", url)

    def remove_webhook(self, url: str) -> None:
        """Unregister a webhook URL."""
        if url in self._webhook_urls:
            self._webhook_urls.remove(url)
            logger.info("Webhook removed: %s", url)

    def add_callback(self, callback: NotificationCallback) -> None:
        """Register an async callback for alert delivery."""
        self._callbacks.append(callback)

    def set_stream_manager(self, manager: Any) -> None:
        """Set or replace the stream manager for WebSocket broadcasting."""
        self._stream_manager = manager

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    async def notify(
        self,
        alerts: list[AnomalyAlert],
    ) -> NotificationSummary:
        """Dispatch alerts through all registered channels.

        Filters by min_severity, then sends to WebSocket, webhooks,
        and callbacks in parallel.

        Args:
            alerts: List of AnomalyAlert entities to dispatch.

        Returns:
            NotificationSummary with per-channel results.
        """
        # Filter by severity
        filtered = [
            a for a in alerts
            if float(a.severity) >= self._min_severity
        ]

        if not filtered:
            return NotificationSummary(total_alerts=0)

        self._stats["total_notifications"] += 1
        self._stats["total_alerts_sent"] += len(filtered)

        summary = NotificationSummary(total_alerts=len(filtered))

        # Dispatch to all channels concurrently
        tasks = []

        if self._stream_manager is not None:
            tasks.append(self._send_websocket(filtered))

        for url in self._webhook_urls:
            tasks.append(self._send_webhook(url, filtered))

        for cb in self._callbacks:
            tasks.append(self._send_callback(cb, filtered))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, NotificationResult):
                    summary.results.append(result)
                elif isinstance(result, Exception):
                    self._stats["errors"] += 1
                    summary.results.append(
                        NotificationResult(
                            channel="unknown",
                            success=False,
                            error=str(result),
                        )
                    )

        return summary

    # ------------------------------------------------------------------
    # Channel: WebSocket/SSE broadcast
    # ------------------------------------------------------------------

    async def _send_websocket(
        self, alerts: list[AnomalyAlert]
    ) -> NotificationResult:
        """Broadcast alerts to WebSocket/SSE clients."""
        import time

        start = time.monotonic()
        total_sent = 0

        try:
            for alert in alerts:
                sent = await self._stream_manager.broadcast_anomaly(
                    symbol=alert.symbol,
                    anomaly_data=_alert_to_dict(alert),
                )
                total_sent += sent

            self._stats["websocket_broadcasts"] += 1
            elapsed = (time.monotonic() - start) * 1000

            return NotificationResult(
                channel="websocket",
                success=True,
                recipients=total_sent,
                latency_ms=round(elapsed, 2),
            )

        except Exception as exc:
            self._stats["errors"] += 1
            elapsed = (time.monotonic() - start) * 1000
            logger.error("WebSocket broadcast failed: %s", exc)
            return NotificationResult(
                channel="websocket",
                success=False,
                error=str(exc),
                latency_ms=round(elapsed, 2),
            )

    # ------------------------------------------------------------------
    # Channel: HTTP webhook
    # ------------------------------------------------------------------

    async def _send_webhook(
        self, url: str, alerts: list[AnomalyAlert]
    ) -> NotificationResult:
        """POST alerts to a webhook URL."""
        import time

        start = time.monotonic()
        payload = {
            "event": "anomaly_alerts",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_alerts": len(alerts),
            "alerts": [_alert_to_dict(a) for a in alerts],
        }

        try:
            async with httpx.AsyncClient(
                timeout=self._webhook_timeout
            ) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-FixTrade-Event": "anomaly_alert",
                    },
                )
                resp.raise_for_status()

            self._stats["webhook_calls"] += 1
            elapsed = (time.monotonic() - start) * 1000

            return NotificationResult(
                channel=f"webhook:{url}",
                success=True,
                recipients=1,
                latency_ms=round(elapsed, 2),
            )

        except Exception as exc:
            self._stats["errors"] += 1
            elapsed = (time.monotonic() - start) * 1000
            logger.error("Webhook POST to %s failed: %s", url, exc)
            return NotificationResult(
                channel=f"webhook:{url}",
                success=False,
                error=str(exc),
                latency_ms=round(elapsed, 2),
            )

    # ------------------------------------------------------------------
    # Channel: Async callbacks
    # ------------------------------------------------------------------

    async def _send_callback(
        self,
        callback: NotificationCallback,
        alerts: list[AnomalyAlert],
    ) -> NotificationResult:
        """Invoke an async callback with the alerts."""
        import time

        start = time.monotonic()

        try:
            await callback(alerts)
            self._stats["callback_invocations"] += 1
            elapsed = (time.monotonic() - start) * 1000

            return NotificationResult(
                channel=f"callback:{callback.__name__}",
                success=True,
                recipients=1,
                latency_ms=round(elapsed, 2),
            )

        except Exception as exc:
            self._stats["errors"] += 1
            elapsed = (time.monotonic() - start) * 1000
            logger.error("Callback %s failed: %s", callback.__name__, exc)
            return NotificationResult(
                channel=f"callback:{callback.__name__}",
                success=False,
                error=str(exc),
                latency_ms=round(elapsed, 2),
            )
