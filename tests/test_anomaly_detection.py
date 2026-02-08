"""
Tests for anomaly detection functionality.

Tests the anomaly detection domain service in isolation.
No database or infrastructure dependencies.
"""

from datetime import date
from decimal import Decimal

import pytest

from app.domain.trading.anomaly_service import AnomalyDetectionService
from app.domain.trading.entities import StockPrice


class TestAnomalyDetectionService:
    """Test suite for the anomaly detection domain service."""

    def test_detect_volume_spike(self):
        """Test detection of volume spikes (>3 std devs)."""
        service = AnomalyDetectionService()

        # Create historical data with normal volume
        prices = [
            StockPrice(
                symbol="TEST",
                date=date(2024, 1, i),
                open=Decimal("10.0"),
                high=Decimal("10.5"),
                low=Decimal("9.5"),
                close=Decimal("10.0"),
                volume=1000,
            )
            for i in range(1, 21)
        ]

        # Add a spike in latest data
        prices.append(
            StockPrice(
                symbol="TEST",
                date=date(2024, 1, 21),
                open=Decimal("10.0"),
                high=Decimal("10.5"),
                low=Decimal("9.5"),
                close=Decimal("10.0"),
                volume=10000,  # 10x normal
            )
        )

        alerts = service.detect_anomalies("TEST", prices)

        assert len(alerts) > 0
        assert any(a.anomaly_type == "volume_spike" for a in alerts)

    def test_detect_price_swing_intraday(self):
        """Test detection of large intraday price swings (>5%)."""
        service = AnomalyDetectionService()

        prices = [
            StockPrice(
                symbol="TEST",
                date=date(2024, 1, i),
                open=Decimal("10.0"),
                high=Decimal("10.2"),
                low=Decimal("9.8"),
                close=Decimal("10.0"),
                volume=1000,
            )
            for i in range(1, 21)
        ]

        # Add a day with large intraday swing
        prices.append(
            StockPrice(
                symbol="TEST",
                date=date(2024, 1, 21),
                open=Decimal("10.0"),
                high=Decimal("11.0"),  # +10%
                low=Decimal("9.0"),  # -10%
                close=Decimal("10.0"),
                volume=1000,
            )
        )

        alerts = service.detect_anomalies("TEST", prices)

        assert len(alerts) > 0
        assert any(a.anomaly_type == "price_swing_intraday" for a in alerts)

    def test_detect_price_swing_daily(self):
        """Test detection of sharp day-over-day price changes (>5%)."""
        service = AnomalyDetectionService()

        prices = [
            StockPrice(
                symbol="TEST",
                date=date(2024, 1, i),
                open=Decimal("10.0"),
                high=Decimal("10.2"),
                low=Decimal("9.8"),
                close=Decimal("10.0"),
                volume=1000,
            )
            for i in range(1, 21)
        ]

        # Add a day with sharp price change
        prices.append(
            StockPrice(
                symbol="TEST",
                date=date(2024, 1, 21),
                open=Decimal("10.0"),
                high=Decimal("11.0"),
                low=Decimal("10.5"),
                close=Decimal("11.0"),  # +10% from previous close
                volume=1000,
            )
        )

        alerts = service.detect_anomalies("TEST", prices)

        assert len(alerts) > 0
        assert any(a.anomaly_type == "price_swing_daily" for a in alerts)

    def test_detect_zero_volume_pattern(self):
        """Test detection of consecutive days with zero volume."""
        service = AnomalyDetectionService()

        prices = [
            StockPrice(
                symbol="TEST",
                date=date(2024, 1, i),
                open=Decimal("10.0"),
                high=Decimal("10.2"),
                low=Decimal("9.8"),
                close=Decimal("10.0"),
                volume=1000 if i < 18 else 0,  # Last 3 days have zero volume
            )
            for i in range(1, 21)
        ]

        alerts = service.detect_anomalies("TEST", prices)

        assert len(alerts) > 0
        assert any(a.anomaly_type == "zero_volume_pattern" for a in alerts)

    def test_detect_price_stagnation(self):
        """Test detection of unusual price stability."""
        service = AnomalyDetectionService()

        # All prices identical for 5 days
        prices = [
            StockPrice(
                symbol="TEST",
                date=date(2024, 1, i),
                open=Decimal("10.0"),
                high=Decimal("10.0"),
                low=Decimal("10.0"),
                close=Decimal("10.0"),
                volume=1000,
            )
            for i in range(1, 21)
        ]

        alerts = service.detect_anomalies("TEST", prices)

        assert len(alerts) > 0
        assert any(a.anomaly_type == "price_stagnation" for a in alerts)

    def test_no_anomalies_with_normal_data(self):
        """Test that normal market data produces no false positives."""
        service = AnomalyDetectionService()

        # Normal, slightly varying prices and volumes
        prices = [
            StockPrice(
                symbol="TEST",
                date=date(2024, 1, i),
                open=Decimal(str(10.0 + (i % 3) * 0.1)),
                high=Decimal(str(10.2 + (i % 3) * 0.1)),
                low=Decimal(str(9.8 + (i % 3) * 0.1)),
                close=Decimal(str(10.0 + (i % 3) * 0.1)),
                volume=1000 + (i % 5) * 100,
            )
            for i in range(1, 21)
        ]

        alerts = service.detect_anomalies("TEST", prices)

        # Should have no significant anomalies
        assert len(alerts) == 0

    def test_insufficient_data(self):
        """Test that insufficient data points return no alerts."""
        service = AnomalyDetectionService()

        # Only 5 data points (minimum is 20)
        prices = [
            StockPrice(
                symbol="TEST",
                date=date(2024, 1, i),
                open=Decimal("10.0"),
                high=Decimal("10.2"),
                low=Decimal("9.8"),
                close=Decimal("10.0"),
                volume=1000,
            )
            for i in range(1, 6)
        ]

        alerts = service.detect_anomalies("TEST", prices)

        assert len(alerts) == 0
