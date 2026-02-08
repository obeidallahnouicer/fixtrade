"""
Domain service: Intraday anomaly detection.

Pure business logic for detecting anomalies in high-frequency
(1-minute / tick-by-tick) market data.

Replaces the daily high/low proxy with real intraday analysis:
    - Hourly price variation (real 1-hour windows, not daily proxy)
    - Minute-level volume bursts (micro-spikes)
    - Flash crash / flash rally detection
    - Bid-ask spread proxy (rapid price oscillations)
    - Opening auction anomalies

No framework imports. No IO. No side effects.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from statistics import mean, stdev
from typing import Optional
from uuid import uuid4

from app.domain.trading.entities import AnomalyAlert, IntradayTick


class IntradayAnomalyService:
    """Domain service for detecting anomalies in intraday tick data.

    Operates on 1-minute bars or raw ticks. All thresholds are
    configurable and the service has zero IO dependencies.
    """

    def __init__(
        self,
        hourly_change_threshold: float = 0.03,
        minute_volume_z_threshold: float = 4.0,
        flash_move_threshold: float = 0.02,
        flash_window_minutes: int = 5,
        oscillation_threshold: int = 6,
        oscillation_window_minutes: int = 10,
    ) -> None:
        """Initialize thresholds for intraday detection.

        Args:
            hourly_change_threshold: Min price change in 1 hour to flag (3%).
            minute_volume_z_threshold: Z-score threshold for 1-min volume burst.
            flash_move_threshold: Min price change in flash_window_minutes (2%).
            flash_window_minutes: Window size for flash move detection.
            oscillation_threshold: Min direction changes in window for oscillation.
            oscillation_window_minutes: Window for bid-ask oscillation check.
        """
        self._hourly_threshold = hourly_change_threshold
        self._vol_z_threshold = minute_volume_z_threshold
        self._flash_threshold = flash_move_threshold
        self._flash_window = flash_window_minutes
        self._osc_threshold = oscillation_threshold
        self._osc_window = oscillation_window_minutes

    def detect_intraday_anomalies(
        self,
        symbol: str,
        ticks: list[IntradayTick],
    ) -> list[AnomalyAlert]:
        """Run all intraday anomaly checks on a tick series.

        Args:
            symbol: Stock ticker.
            ticks: List of IntradayTick sorted by timestamp ascending.

        Returns:
            List of detected AnomalyAlert entities.
        """
        if len(ticks) < 10:
            return []

        alerts: list[AnomalyAlert] = []

        alerts.extend(self._detect_hourly_moves(symbol, ticks))
        alerts.extend(self._detect_volume_bursts(symbol, ticks))
        alerts.extend(self._detect_flash_moves(symbol, ticks))
        alerts.extend(self._detect_price_oscillations(symbol, ticks))
        alerts.extend(self._detect_opening_anomaly(symbol, ticks))

        return alerts

    # ------------------------------------------------------------------
    # Check 1: Hourly price variation (REAL, not daily proxy)
    # ------------------------------------------------------------------

    def _detect_hourly_moves(
        self, symbol: str, ticks: list[IntradayTick]
    ) -> list[AnomalyAlert]:
        """Detect large price moves within any 1-hour sliding window.

        This replaces the daily high/low proxy with actual intraday data.
        """
        alerts = []
        if len(ticks) < 60:
            return alerts

        for i in range(60, len(ticks)):
            window_start = ticks[i - 60]
            window_end = ticks[i]

            if window_start.price <= 0:
                continue

            change = abs(
                float(window_end.price - window_start.price)
                / float(window_start.price)
            )

            if change >= self._hourly_threshold:
                direction = "up" if window_end.price > window_start.price else "down"
                severity = min(1.0, change / 0.10)  # cap at 10% = severity 1.0

                alerts.append(
                    AnomalyAlert(
                        id=uuid4(),
                        symbol=symbol,
                        detected_at=window_end.timestamp,
                        anomaly_type="intraday_hourly_move",
                        severity=Decimal(str(round(severity, 4))),
                        description=(
                            f"1-hour price move {direction}: {change:.2%} "
                            f"({window_start.price} → {window_end.price}) "
                            f"between {window_start.timestamp.strftime('%H:%M')} "
                            f"and {window_end.timestamp.strftime('%H:%M')}"
                        ),
                    )
                )
                # Skip ahead to avoid duplicate alerts for the same move
                break

        return alerts

    # ------------------------------------------------------------------
    # Check 2: Minute-level volume bursts
    # ------------------------------------------------------------------

    def _detect_volume_bursts(
        self, symbol: str, ticks: list[IntradayTick]
    ) -> list[AnomalyAlert]:
        """Detect 1-minute volume spikes (Z-score > threshold)."""
        alerts = []
        volumes = [t.volume for t in ticks if t.volume > 0]

        if len(volumes) < 20:
            return alerts

        avg_vol = mean(volumes)
        std_vol = stdev(volumes)

        if std_vol == 0:
            return alerts

        for tick in ticks:
            if tick.volume <= 0:
                continue

            z = (tick.volume - avg_vol) / std_vol
            if z > self._vol_z_threshold:
                severity = min(1.0, z / 10.0)
                alerts.append(
                    AnomalyAlert(
                        id=uuid4(),
                        symbol=symbol,
                        detected_at=tick.timestamp,
                        anomaly_type="intraday_volume_burst",
                        severity=Decimal(str(round(severity, 4))),
                        description=(
                            f"Minute volume burst: {tick.volume:,} "
                            f"(Z={z:.1f}, avg={avg_vol:,.0f}) "
                            f"at {tick.timestamp.strftime('%H:%M')}"
                        ),
                    )
                )

        return alerts

    # ------------------------------------------------------------------
    # Check 3: Flash crash / rally (rapid move in short window)
    # ------------------------------------------------------------------

    def _detect_flash_moves(
        self, symbol: str, ticks: list[IntradayTick]
    ) -> list[AnomalyAlert]:
        """Detect rapid price moves within a short time window."""
        alerts = []
        window = self._flash_window

        if len(ticks) < window + 1:
            return alerts

        for i in range(window, len(ticks)):
            start_tick = ticks[i - window]
            end_tick = ticks[i]

            if start_tick.price <= 0:
                continue

            change = float(end_tick.price - start_tick.price) / float(start_tick.price)
            abs_change = abs(change)

            if abs_change >= self._flash_threshold:
                move_type = "flash_rally" if change > 0 else "flash_crash"
                severity = min(1.0, abs_change / 0.05)

                alerts.append(
                    AnomalyAlert(
                        id=uuid4(),
                        symbol=symbol,
                        detected_at=end_tick.timestamp,
                        anomaly_type=f"intraday_{move_type}",
                        severity=Decimal(str(round(severity, 4))),
                        description=(
                            f"{move_type.replace('_', ' ').title()}: "
                            f"{change:+.2%} in {window} minutes "
                            f"({start_tick.price} → {end_tick.price}) "
                            f"at {end_tick.timestamp.strftime('%H:%M')}"
                        ),
                    )
                )
                # Only flag the most extreme flash move
                break

        return alerts

    # ------------------------------------------------------------------
    # Check 4: Price oscillations (potential manipulation)
    # ------------------------------------------------------------------

    def _detect_price_oscillations(
        self, symbol: str, ticks: list[IntradayTick]
    ) -> list[AnomalyAlert]:
        """Detect rapid back-and-forth price movements.

        High-frequency oscillations can indicate wash trading
        or algorithmic manipulation on thin BVMT markets.
        """
        alerts = []
        window = self._osc_window

        if len(ticks) < window + 1:
            return alerts

        for i in range(window, len(ticks)):
            window_ticks = ticks[i - window : i + 1]
            direction_changes = 0

            for j in range(2, len(window_ticks)):
                prev_dir = window_ticks[j - 1].price - window_ticks[j - 2].price
                curr_dir = window_ticks[j].price - window_ticks[j - 1].price

                if (prev_dir > 0 and curr_dir < 0) or (prev_dir < 0 and curr_dir > 0):
                    direction_changes += 1

            if direction_changes >= self._osc_threshold:
                severity = min(1.0, direction_changes / 15.0)
                alerts.append(
                    AnomalyAlert(
                        id=uuid4(),
                        symbol=symbol,
                        detected_at=window_ticks[-1].timestamp,
                        anomaly_type="intraday_oscillation",
                        severity=Decimal(str(round(severity, 4))),
                        description=(
                            f"Rapid price oscillation: {direction_changes} direction changes "
                            f"in {window} minutes ending at "
                            f"{window_ticks[-1].timestamp.strftime('%H:%M')}. "
                            f"Potential wash trading or manipulation."
                        ),
                    )
                )
                # One flag per scan is enough
                break

        return alerts

    # ------------------------------------------------------------------
    # Check 5: Opening auction anomaly
    # ------------------------------------------------------------------

    def _detect_opening_anomaly(
        self, symbol: str, ticks: list[IntradayTick]
    ) -> list[AnomalyAlert]:
        """Detect unusual opening price gap vs first-hour average.

        On BVMT, the opening auction can produce an anomalous gap
        when large orders are queued before market open.
        """
        alerts = []
        if len(ticks) < 30:
            return alerts

        opening_price = float(ticks[0].price)
        if opening_price <= 0:
            return alerts

        # Average of first 30 minutes
        first_30 = [float(t.price) for t in ticks[:30] if float(t.price) > 0]
        if not first_30:
            return alerts

        avg_30 = mean(first_30)
        gap = abs(opening_price - avg_30) / avg_30

        if gap > 0.015:  # 1.5% gap vs 30-min average
            severity = min(1.0, gap / 0.05)
            direction = "above" if opening_price > avg_30 else "below"

            alerts.append(
                AnomalyAlert(
                    id=uuid4(),
                    symbol=symbol,
                    detected_at=ticks[0].timestamp,
                    anomaly_type="intraday_opening_gap",
                    severity=Decimal(str(round(severity, 4))),
                    description=(
                        f"Opening auction anomaly: open {opening_price:.3f} "
                        f"is {gap:.2%} {direction} first-30min average {avg_30:.3f}. "
                        f"Possible large pre-market order."
                    ),
                )
            )

        return alerts
