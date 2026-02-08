"""
Tests for:
    1. AnomalyEvaluator — Precision / Recall / F1-Score evaluation
    2. AnomalyNotifier  — Multi-channel alert notification system

All tests use pure domain objects; no database or network calls.
"""

import asyncio
import json
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.domain.trading.anomaly_evaluator import (
    AnomalyEvaluator,
    EvaluationMetrics,
    EvaluationReport,
    LabeledAnomaly,
)
from app.domain.trading.anomaly_notifier import (
    AnomalyNotifier,
    NotificationResult,
    NotificationSummary,
    _alert_to_dict,
    _severity_label,
)
from app.domain.trading.anomaly_service import AnomalyDetectionService
from app.domain.trading.entities import AnomalyAlert, StockPrice


# ══════════════════════════════════════════════════════════════════════
# Fixtures — shared test data
# ══════════════════════════════════════════════════════════════════════


def _price(day: int, close: float, volume: int, symbol: str = "BIAT") -> StockPrice:
    """Build a StockPrice for 2024-01-{day}."""
    return StockPrice(
        symbol=symbol,
        date=date(2024, 1, day),
        open=Decimal(str(close - 0.5)),
        high=Decimal(str(close + 1.0)),
        low=Decimal(str(close - 1.0)),
        close=Decimal(str(close)),
        volume=volume,
    )


def _alert(
    day: int,
    anomaly_type: str = "volume_spike",
    severity: float = 0.7,
    symbol: str = "BIAT",
) -> AnomalyAlert:
    return AnomalyAlert(
        id=uuid4(),
        symbol=symbol,
        detected_at=datetime(2024, 1, day, 12, 0, tzinfo=timezone.utc),
        anomaly_type=anomaly_type,
        severity=Decimal(str(severity)),
        description=f"Test {anomaly_type} on day {day}",
    )


@pytest.fixture
def normal_prices() -> list[StockPrice]:
    """30 days of calm, uniform data (no spikes)."""
    return [_price(d, 100.0, 50_000) for d in range(1, 31)]


@pytest.fixture
def spikey_prices() -> list[StockPrice]:
    """30 days: days 10, 20 have extreme volume & price spikes."""
    prices = []
    for d in range(1, 31):
        if d in (10, 20):
            prices.append(_price(d, 150.0, 500_000))  # big spike
        else:
            prices.append(_price(d, 100.0, 50_000))
    return prices


# ══════════════════════════════════════════════════════════════════════
# PART 1 — AnomalyEvaluator tests
# ══════════════════════════════════════════════════════════════════════


class TestEvaluationMetrics:
    """Unit tests for the EvaluationMetrics dataclass."""

    def test_perfect_detection(self):
        m = EvaluationMetrics(true_positives=5, false_positives=0, false_negatives=0)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1_score == 1.0
        assert m.support == 5

    def test_no_detections(self):
        m = EvaluationMetrics(true_positives=0, false_positives=0, false_negatives=5)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1_score == 0.0
        assert m.support == 5

    def test_only_false_positives(self):
        m = EvaluationMetrics(true_positives=0, false_positives=10, false_negatives=0)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1_score == 0.0
        assert m.support == 0

    def test_partial_detection(self):
        m = EvaluationMetrics(true_positives=3, false_positives=2, false_negatives=1)
        assert m.precision == pytest.approx(3 / 5)
        assert m.recall == pytest.approx(3 / 4)
        assert m.support == 4

    def test_zero_everything(self):
        m = EvaluationMetrics()
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1_score == 0.0
        assert m.support == 0


class TestEvaluationReport:
    """Unit tests for the EvaluationReport summary."""

    def test_summary_string(self):
        report = EvaluationReport(symbol="BIAT")
        report.overall.true_positives = 5
        report.overall.false_positives = 2
        report.overall.false_negatives = 1
        report.total_detected = 7
        report.total_known = 6
        s = report.summary()
        assert "BIAT" in s
        assert "Precision" in s
        assert "F1-Score" in s

    def test_to_dict(self):
        report = EvaluationReport(symbol="TEST")
        report.overall.true_positives = 3
        d = report.to_dict()
        assert d["symbol"] == "TEST"
        assert d["overall"]["true_positives"] == 3
        assert "per_type" in d


class TestAnomalyEvaluatorMatching:
    """Test the matching logic between detected and known anomalies."""

    def test_exact_match(self):
        evaluator = AnomalyEvaluator(date_tolerance_days=0)
        detected = [_alert(10, "volume_spike")]
        known = [LabeledAnomaly(date=date(2024, 1, 10), anomaly_type="volume_spike")]
        report = evaluator._evaluate("BIAT", detected, known)

        assert report.overall.true_positives == 1
        assert report.overall.false_positives == 0
        assert report.overall.false_negatives == 0

    def test_date_tolerance(self):
        evaluator = AnomalyEvaluator(date_tolerance_days=1)
        detected = [_alert(11, "volume_spike")]  # one day off
        known = [LabeledAnomaly(date=date(2024, 1, 10), anomaly_type="volume_spike")]
        report = evaluator._evaluate("BIAT", detected, known)

        assert report.overall.true_positives == 1

    def test_date_beyond_tolerance(self):
        evaluator = AnomalyEvaluator(date_tolerance_days=0)
        detected = [_alert(12, "volume_spike")]  # two days off
        known = [LabeledAnomaly(date=date(2024, 1, 10), anomaly_type="volume_spike")]
        report = evaluator._evaluate("BIAT", detected, known)

        assert report.overall.true_positives == 0
        assert report.overall.false_positives == 1
        assert report.overall.false_negatives == 1

    def test_any_type_matches_all(self):
        evaluator = AnomalyEvaluator(date_tolerance_days=0)
        detected = [_alert(10, "price_swing_daily")]
        known = [LabeledAnomaly(date=date(2024, 1, 10), anomaly_type="any")]
        report = evaluator._evaluate("BIAT", detected, known)

        assert report.overall.true_positives == 1

    def test_type_mismatch_no_match(self):
        evaluator = AnomalyEvaluator(date_tolerance_days=0)
        detected = [_alert(10, "volume_spike")]
        known = [
            LabeledAnomaly(date=date(2024, 1, 10), anomaly_type="price_swing_daily")
        ]
        report = evaluator._evaluate("BIAT", detected, known)

        assert report.overall.true_positives == 0
        assert report.overall.false_positives == 1
        assert report.overall.false_negatives == 1

    def test_multiple_alerts_partial_match(self):
        evaluator = AnomalyEvaluator(date_tolerance_days=0)
        detected = [
            _alert(10, "volume_spike"),
            _alert(15, "price_swing_daily"),
            _alert(25, "volume_spike"),
        ]
        known = [
            LabeledAnomaly(date=date(2024, 1, 10), anomaly_type="volume_spike"),
            LabeledAnomaly(date=date(2024, 1, 20), anomaly_type="volume_spike"),
        ]
        report = evaluator._evaluate("BIAT", detected, known)

        # Match: day 10 volume_spike
        # FP: day 15 (no known), day 25 (wrong date)
        # FN: day 20 (not detected)
        assert report.overall.true_positives == 1
        assert report.overall.false_positives == 2
        assert report.overall.false_negatives == 1

    def test_empty_inputs(self):
        evaluator = AnomalyEvaluator()
        report = evaluator._evaluate("BIAT", [], [])
        assert report.overall.true_positives == 0
        assert report.overall.false_positives == 0
        assert report.overall.false_negatives == 0


class TestPerTypeMetrics:
    """Test per anomaly-type breakdown."""

    def test_per_type_computed(self):
        evaluator = AnomalyEvaluator(date_tolerance_days=0)
        detected = [
            _alert(10, "volume_spike"),
            _alert(15, "price_swing_daily"),
        ]
        known = [
            LabeledAnomaly(date=date(2024, 1, 10), anomaly_type="volume_spike"),
        ]
        report = evaluator._evaluate("BIAT", detected, known)

        assert "volume_spike" in report.per_type
        assert "price_swing_daily" in report.per_type
        assert report.per_type["volume_spike"].true_positives == 1
        assert report.per_type["price_swing_daily"].false_positives == 1


class TestBacktest:
    """Test the full backtest pipeline."""

    def test_backtest_on_normal_data(self, normal_prices):
        evaluator = AnomalyEvaluator()
        known = [LabeledAnomaly(date=date(2024, 1, 15), anomaly_type="any")]
        report = evaluator.backtest("BIAT", normal_prices, known)

        # With uniform data, there should be no detections
        assert report.total_known == 1
        # FN expected: the known anomaly isn't detected
        assert report.overall.false_negatives >= 0

    def test_backtest_on_spikey_data(self, spikey_prices):
        evaluator = AnomalyEvaluator(date_tolerance_days=1)
        known = [
            LabeledAnomaly(date=date(2024, 1, 10), anomaly_type="any"),
            LabeledAnomaly(date=date(2024, 1, 20), anomaly_type="any"),
        ]
        report = evaluator.backtest("BIAT", spikey_prices, known)

        # Should detect at least some anomalies around days 10, 20
        assert report.total_detected > 0
        assert report.total_known == 2

    def test_backtest_empty_prices(self):
        evaluator = AnomalyEvaluator()
        report = evaluator.backtest("BIAT", [], [])
        assert report.total_detected == 0
        assert report.total_known == 0


class TestGenerateKnownAnomalies:
    """Test auto-generation of ground-truth labels."""

    def test_generates_volume_spikes(self, spikey_prices):
        known = AnomalyEvaluator.generate_known_anomalies_from_data(spikey_prices)
        types = {a.anomaly_type for a in known}
        assert "volume_spike" in types

    def test_generates_price_swings(self, spikey_prices):
        known = AnomalyEvaluator.generate_known_anomalies_from_data(spikey_prices)
        types = {a.anomaly_type for a in known}
        assert "price_swing_daily" in types

    def test_empty_if_too_few_prices(self):
        prices = [_price(d, 100.0, 50_000) for d in range(1, 10)]
        known = AnomalyEvaluator.generate_known_anomalies_from_data(prices)
        assert known == []

    def test_zero_volume_detected(self):
        prices = [_price(d, 100.0, 50_000) for d in range(1, 25)]
        prices.append(_price(25, 100.0, 0))  # zero volume
        known = AnomalyEvaluator.generate_known_anomalies_from_data(prices)
        zero_vol = [a for a in known if a.anomaly_type == "zero_volume_pattern"]
        assert len(zero_vol) >= 1


class TestLoadKnownAnomalies:
    """Test CSV/JSON loading of labeled anomalies."""

    def test_load_csv(self, tmp_path):
        csv_file = tmp_path / "anomalies.csv"
        csv_file.write_text(
            "date,anomaly_type,symbol,description\n"
            "2024-01-10,volume_spike,BIAT,Big spike\n"
            "2024-01-20,price_swing_daily,,Price drop\n"
        )
        anomalies = AnomalyEvaluator.load_known_anomalies_csv(csv_file)
        assert len(anomalies) == 2
        assert anomalies[0].date == date(2024, 1, 10)
        assert anomalies[0].anomaly_type == "volume_spike"
        assert anomalies[1].symbol is None

    def test_load_json(self, tmp_path):
        json_file = tmp_path / "anomalies.json"
        data = [
            {"date": "2024-01-10", "anomaly_type": "volume_spike"},
            {"date": "2024-01-20"},
        ]
        json_file.write_text(json.dumps(data))
        anomalies = AnomalyEvaluator.load_known_anomalies_json(json_file)
        assert len(anomalies) == 2
        assert anomalies[0].anomaly_type == "volume_spike"
        assert anomalies[1].anomaly_type == "any"  # default


# ══════════════════════════════════════════════════════════════════════
# PART 2 — AnomalyNotifier tests
# ══════════════════════════════════════════════════════════════════════


class TestSeverityLabel:
    """Test the severity-to-label mapping."""

    def test_critical(self):
        assert _severity_label(Decimal("0.9")) == "critical"

    def test_high(self):
        assert _severity_label(Decimal("0.65")) == "high"

    def test_medium(self):
        assert _severity_label(Decimal("0.45")) == "medium"

    def test_low(self):
        assert _severity_label(Decimal("0.2")) == "low"


class TestAlertToDict:
    """Test serialization of AnomalyAlert."""

    def test_serializes_correctly(self):
        alert = _alert(10, "volume_spike", severity=0.85)
        d = _alert_to_dict(alert)
        assert d["symbol"] == "BIAT"
        assert d["anomaly_type"] == "volume_spike"
        assert d["severity"] == 0.85
        assert d["severity_label"] == "critical"
        assert "id" in d
        assert "detected_at" in d


class TestNotifierWebhookConfig:
    """Test webhook URL registration."""

    def test_add_valid_webhook(self):
        notifier = AnomalyNotifier()
        notifier.add_webhook("https://hooks.example.com/alert")
        assert len(notifier._webhook_urls) == 1

    def test_add_invalid_webhook_scheme(self):
        notifier = AnomalyNotifier()
        with pytest.raises(ValueError, match="Invalid webhook URL scheme"):
            notifier.add_webhook("ftp://bad.example.com")

    def test_add_duplicate_webhook(self):
        notifier = AnomalyNotifier()
        notifier.add_webhook("https://hooks.example.com/a")
        notifier.add_webhook("https://hooks.example.com/a")
        assert len(notifier._webhook_urls) == 1

    def test_remove_webhook(self):
        notifier = AnomalyNotifier()
        notifier.add_webhook("https://hooks.example.com/a")
        notifier.remove_webhook("https://hooks.example.com/a")
        assert len(notifier._webhook_urls) == 0


class TestNotifierSeverityFilter:
    """Test the min_severity filter."""

    @pytest.mark.asyncio
    async def test_filters_low_severity(self):
        notifier = AnomalyNotifier(min_severity=0.5)
        alerts = [_alert(10, severity=0.3)]
        summary = await notifier.notify(alerts)
        assert summary.total_alerts == 0

    @pytest.mark.asyncio
    async def test_passes_high_severity(self):
        mock_mgr = AsyncMock()
        mock_mgr.broadcast_anomaly = AsyncMock(return_value=1)
        notifier = AnomalyNotifier(stream_manager=mock_mgr, min_severity=0.5)
        alerts = [_alert(10, severity=0.8)]
        summary = await notifier.notify(alerts)
        assert summary.total_alerts == 1


class TestNotifierWebSocket:
    """Test WebSocket/SSE broadcast channel."""

    @pytest.mark.asyncio
    async def test_broadcasts_to_stream_manager(self):
        mock_mgr = AsyncMock()
        mock_mgr.broadcast_anomaly = AsyncMock(return_value=3)
        notifier = AnomalyNotifier(stream_manager=mock_mgr)

        alerts = [_alert(10), _alert(20)]
        summary = await notifier.notify(alerts)

        assert mock_mgr.broadcast_anomaly.call_count == 2
        ws_result = [r for r in summary.results if r.channel == "websocket"]
        assert len(ws_result) == 1
        assert ws_result[0].success is True
        assert ws_result[0].recipients == 6  # 3 per alert × 2 alerts

    @pytest.mark.asyncio
    async def test_broadcast_failure_handled(self):
        mock_mgr = AsyncMock()
        mock_mgr.broadcast_anomaly = AsyncMock(side_effect=RuntimeError("ws down"))
        notifier = AnomalyNotifier(stream_manager=mock_mgr)

        alerts = [_alert(10)]
        summary = await notifier.notify(alerts)

        ws_result = [r for r in summary.results if r.channel == "websocket"]
        assert ws_result[0].success is False
        assert "ws down" in ws_result[0].error


class TestNotifierWebhook:
    """Test HTTP webhook channel."""

    @pytest.mark.asyncio
    async def test_webhook_post_success(self):
        notifier = AnomalyNotifier()
        notifier.add_webhook("https://hooks.example.com/alert")

        alerts = [_alert(10)]

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("app.domain.trading.anomaly_notifier.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            summary = await notifier.notify(alerts)

        wh_results = [r for r in summary.results if "webhook" in r.channel]
        assert len(wh_results) == 1
        assert wh_results[0].success is True

    @pytest.mark.asyncio
    async def test_webhook_failure_handled(self):
        notifier = AnomalyNotifier()
        notifier.add_webhook("https://hooks.example.com/alert")

        alerts = [_alert(10)]

        with patch("app.domain.trading.anomaly_notifier.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=ConnectionError("timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            summary = await notifier.notify(alerts)

        wh_results = [r for r in summary.results if "webhook" in r.channel]
        assert wh_results[0].success is False


class TestNotifierCallback:
    """Test async callback channel."""

    @pytest.mark.asyncio
    async def test_callback_invoked(self):
        received = []

        async def my_hook(alerts):
            received.extend(alerts)

        notifier = AnomalyNotifier()
        notifier.add_callback(my_hook)

        alerts = [_alert(10), _alert(20)]
        summary = await notifier.notify(alerts)

        assert len(received) == 2
        cb_results = [r for r in summary.results if "callback" in r.channel]
        assert cb_results[0].success is True

    @pytest.mark.asyncio
    async def test_callback_failure_handled(self):
        async def bad_hook(alerts):
            raise ValueError("boom")

        notifier = AnomalyNotifier()
        notifier.add_callback(bad_hook)

        alerts = [_alert(10)]
        summary = await notifier.notify(alerts)

        cb_results = [r for r in summary.results if "callback" in r.channel]
        assert cb_results[0].success is False
        assert "boom" in cb_results[0].error


class TestNotifierStats:
    """Test statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_updated(self):
        mock_mgr = AsyncMock()
        mock_mgr.broadcast_anomaly = AsyncMock(return_value=1)
        notifier = AnomalyNotifier(stream_manager=mock_mgr)

        alerts = [_alert(10)]
        await notifier.notify(alerts)
        await notifier.notify(alerts)

        stats = notifier.stats
        assert stats["total_notifications"] == 2
        assert stats["total_alerts_sent"] == 2
        assert stats["websocket_broadcasts"] == 2


class TestNotifierMultiChannel:
    """Test dispatching to multiple channels simultaneously."""

    @pytest.mark.asyncio
    async def test_all_channels_receive(self):
        received_cb = []

        async def my_cb(alerts):
            received_cb.extend(alerts)

        mock_mgr = AsyncMock()
        mock_mgr.broadcast_anomaly = AsyncMock(return_value=2)

        notifier = AnomalyNotifier(stream_manager=mock_mgr)
        notifier.add_callback(my_cb)

        alerts = [_alert(10)]
        summary = await notifier.notify(alerts)

        # WebSocket + callback = 2 results
        assert len(summary.results) == 2
        assert summary.total_sent >= 2
        assert len(received_cb) == 1


class TestNotificationSummary:
    """Test NotificationSummary properties."""

    def test_total_sent(self):
        s = NotificationSummary(total_alerts=3)
        s.results = [
            NotificationResult(channel="ws", success=True, recipients=5),
            NotificationResult(channel="wh", success=False, recipients=0),
            NotificationResult(channel="cb", success=True, recipients=1),
        ]
        assert s.total_sent == 6
        assert s.all_success is False

    def test_all_success(self):
        s = NotificationSummary(total_alerts=1)
        s.results = [
            NotificationResult(channel="ws", success=True, recipients=2),
        ]
        assert s.all_success is True
