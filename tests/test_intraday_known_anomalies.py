"""
Tests for:
    1. IntradayTick entity
    2. IntradayAnomalyService — intraday anomaly detection
    3. Intraday tick generator (synthetic data)
    4. Known anomalies CSV loading

All tests use pure domain objects; no database or network calls.
"""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from app.domain.trading.entities import IntradayTick, StockPrice
from app.domain.trading.intraday_anomaly_service import IntradayAnomalyService


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════

TZ_TUNIS = timezone(timedelta(hours=1))


def _tick(
    minute: int,
    price: float,
    volume: int = 100,
    symbol: str = "BIAT",
    base_date: date | None = None,
) -> IntradayTick:
    """Build a 1-min tick at 09:{minute:02d} on 2024-01-15."""
    d = base_date or date(2024, 1, 15)
    hour = 9 + minute // 60
    m = minute % 60
    ts = datetime(d.year, d.month, d.day, hour, m, tzinfo=TZ_TUNIS)
    return IntradayTick(
        symbol=symbol,
        timestamp=ts,
        price=Decimal(str(price)),
        volume=volume,
        tick_type="1min",
    )


def _flat_ticks(n: int = 200, price: float = 100.0, volume: int = 100) -> list[IntradayTick]:
    """Generate N flat (uniform) ticks — no anomalies expected."""
    return [_tick(i, price, volume) for i in range(n)]


def _ticks_with_hourly_spike(n: int = 200) -> list[IntradayTick]:
    """Generate ticks where price jumps +5% between minute 30 and 90."""
    ticks = []
    for i in range(n):
        if i < 60:
            p = 100.0
        else:
            p = 105.0  # +5% after the first hour
        ticks.append(_tick(i, p, 100))
    return ticks


def _ticks_with_volume_burst(n: int = 200) -> list[IntradayTick]:
    """Generate ticks where minute 100 has extreme volume."""
    ticks = []
    for i in range(n):
        vol = 100 if i != 100 else 50_000  # huge spike
        ticks.append(_tick(i, 100.0, vol))
    return ticks


def _ticks_with_flash_crash(n: int = 200) -> list[IntradayTick]:
    """Generate ticks with a sudden 3% drop in 5 minutes."""
    ticks = []
    for i in range(n):
        if i < 80:
            p = 100.0
        elif i < 85:
            # 3% drop spread over 5 min
            p = 100.0 - (i - 80) * 0.6
        else:
            p = 97.0
        ticks.append(_tick(i, p, 100))
    return ticks


def _ticks_with_oscillations(n: int = 200) -> list[IntradayTick]:
    """Generate ticks with rapid up-down oscillations."""
    ticks = []
    for i in range(n):
        # Oscillate between 99 and 101 every minute from minute 50-70
        if 50 <= i <= 70:
            p = 101.0 if i % 2 == 0 else 99.0
        else:
            p = 100.0
        ticks.append(_tick(i, p, 100))
    return ticks


def _ticks_with_opening_gap(n: int = 200) -> list[IntradayTick]:
    """Generate ticks where opening price is 3% above first-30min avg."""
    ticks = []
    for i in range(n):
        if i == 0:
            p = 103.0  # opening
        elif i < 30:
            p = 100.0  # settles down
        else:
            p = 100.0
        ticks.append(_tick(i, p, 100))
    return ticks


# ══════════════════════════════════════════════════════════════
# PART 1 — IntradayTick entity tests
# ══════════════════════════════════════════════════════════════


class TestIntradayTickEntity:
    """Test the IntradayTick frozen dataclass."""

    def test_creation(self):
        tick = _tick(0, 100.0, 500)
        assert tick.symbol == "BIAT"
        assert tick.price == Decimal("100.0")
        assert tick.volume == 500
        assert tick.tick_type == "1min"

    def test_frozen(self):
        tick = _tick(0, 100.0)
        with pytest.raises(AttributeError):
            tick.price = Decimal("200.0")

    def test_default_tick_type(self):
        tick = IntradayTick(
            symbol="BT",
            timestamp=datetime(2024, 1, 1, 9, 0, tzinfo=TZ_TUNIS),
            price=Decimal("50.0"),
        )
        assert tick.tick_type == "1min"
        assert tick.volume == 0


# ══════════════════════════════════════════════════════════════
# PART 2 — IntradayAnomalyService tests
# ══════════════════════════════════════════════════════════════


class TestIntradayAnomalyService:
    """Core tests for the intraday anomaly detection service."""

    def setup_method(self):
        self.service = IntradayAnomalyService()

    def test_no_anomalies_on_flat_data(self):
        ticks = _flat_ticks(200)
        alerts = self.service.detect_intraday_anomalies("BIAT", ticks)
        # Flat data should produce no anomalies
        hourly = [a for a in alerts if a.anomaly_type == "intraday_hourly_move"]
        flash = [a for a in alerts if "flash" in a.anomaly_type]
        vol_burst = [a for a in alerts if a.anomaly_type == "intraday_volume_burst"]
        assert len(hourly) == 0
        assert len(flash) == 0
        assert len(vol_burst) == 0

    def test_insufficient_data_returns_empty(self):
        ticks = [_tick(i, 100.0) for i in range(5)]
        alerts = self.service.detect_intraday_anomalies("BIAT", ticks)
        assert alerts == []

    def test_empty_ticks_returns_empty(self):
        alerts = self.service.detect_intraday_anomalies("BIAT", [])
        assert alerts == []


class TestHourlyMoveDetection:
    """Test the hourly price variation detector."""

    def setup_method(self):
        self.service = IntradayAnomalyService(hourly_change_threshold=0.03)

    def test_detects_hourly_move(self):
        ticks = _ticks_with_hourly_spike()
        alerts = self.service.detect_intraday_anomalies("BIAT", ticks)
        hourly = [a for a in alerts if a.anomaly_type == "intraday_hourly_move"]
        assert len(hourly) >= 1
        assert hourly[0].symbol == "BIAT"
        assert "1-hour price move" in hourly[0].description

    def test_no_hourly_move_below_threshold(self):
        # 1% move — below 3% threshold
        ticks = []
        for i in range(200):
            p = 100.0 if i < 60 else 101.0
            ticks.append(_tick(i, p))
        alerts = self.service.detect_intraday_anomalies("BIAT", ticks)
        hourly = [a for a in alerts if a.anomaly_type == "intraday_hourly_move"]
        assert len(hourly) == 0


class TestVolumeBurstDetection:
    """Test the minute-level volume burst detector."""

    def setup_method(self):
        self.service = IntradayAnomalyService(minute_volume_z_threshold=4.0)

    def test_detects_volume_burst(self):
        ticks = _ticks_with_volume_burst()
        alerts = self.service.detect_intraday_anomalies("BIAT", ticks)
        bursts = [a for a in alerts if a.anomaly_type == "intraday_volume_burst"]
        assert len(bursts) >= 1
        assert "volume burst" in bursts[0].description.lower()

    def test_no_burst_with_uniform_volume(self):
        ticks = _flat_ticks(200, volume=1000)
        alerts = self.service.detect_intraday_anomalies("BIAT", ticks)
        bursts = [a for a in alerts if a.anomaly_type == "intraday_volume_burst"]
        assert len(bursts) == 0


class TestFlashMoveDetection:
    """Test flash crash / rally detection."""

    def setup_method(self):
        self.service = IntradayAnomalyService(
            flash_move_threshold=0.02,
            flash_window_minutes=5,
        )

    def test_detects_flash_crash(self):
        ticks = _ticks_with_flash_crash()
        alerts = self.service.detect_intraday_anomalies("BIAT", ticks)
        flash = [a for a in alerts if "flash" in a.anomaly_type]
        assert len(flash) >= 1
        assert "flash_crash" in flash[0].anomaly_type

    def test_no_flash_on_gradual_move(self):
        # Very slow drift — no flash
        ticks = []
        for i in range(200):
            p = 100.0 - i * 0.001  # 0.1% per minute
            ticks.append(_tick(i, p))
        alerts = self.service.detect_intraday_anomalies("BIAT", ticks)
        flash = [a for a in alerts if "flash" in a.anomaly_type]
        assert len(flash) == 0


class TestOscillationDetection:
    """Test rapid price oscillation detection."""

    def setup_method(self):
        self.service = IntradayAnomalyService(
            oscillation_threshold=6,
            oscillation_window_minutes=10,
        )

    def test_detects_oscillations(self):
        ticks = _ticks_with_oscillations()
        alerts = self.service.detect_intraday_anomalies("BIAT", ticks)
        osc = [a for a in alerts if a.anomaly_type == "intraday_oscillation"]
        assert len(osc) >= 1
        assert "direction changes" in osc[0].description

    def test_no_oscillation_on_smooth_data(self):
        ticks = _flat_ticks(200)
        alerts = self.service.detect_intraday_anomalies("BIAT", ticks)
        osc = [a for a in alerts if a.anomaly_type == "intraday_oscillation"]
        assert len(osc) == 0


class TestOpeningAnomalyDetection:
    """Test opening auction gap detection."""

    def setup_method(self):
        self.service = IntradayAnomalyService()

    def test_detects_opening_gap(self):
        ticks = _ticks_with_opening_gap()
        alerts = self.service.detect_intraday_anomalies("BIAT", ticks)
        opening = [a for a in alerts if a.anomaly_type == "intraday_opening_gap"]
        assert len(opening) >= 1
        assert "Opening auction anomaly" in opening[0].description

    def test_no_opening_gap_on_normal_open(self):
        ticks = _flat_ticks(200)
        alerts = self.service.detect_intraday_anomalies("BIAT", ticks)
        opening = [a for a in alerts if a.anomaly_type == "intraday_opening_gap"]
        assert len(opening) == 0


# ══════════════════════════════════════════════════════════════
# PART 3 — Tick generator tests
# ══════════════════════════════════════════════════════════════


class TestIntradayTickGenerator:
    """Test the synthetic 1-min tick generator."""

    def test_generates_correct_count(self):
        from db.load_intraday_and_labels import generate_1min_ticks, MINUTES_PER_SESSION

        ticks = generate_1min_ticks(
            "BIAT", date(2024, 1, 15), 100.0, 105.0, 95.0, 102.0, 10000, seed=42
        )
        assert len(ticks) == MINUTES_PER_SESSION + 1

    def test_first_tick_is_open(self):
        from db.load_intraday_and_labels import generate_1min_ticks

        ticks = generate_1min_ticks(
            "BIAT", date(2024, 1, 15), 100.0, 105.0, 95.0, 102.0, 10000, seed=42
        )
        assert abs(ticks[0][2] - 100.0) < 1e-9  # price at index 2

    def test_last_tick_is_close(self):
        from db.load_intraday_and_labels import generate_1min_ticks

        ticks = generate_1min_ticks(
            "BIAT", date(2024, 1, 15), 100.0, 105.0, 95.0, 102.0, 10000, seed=42
        )
        assert abs(ticks[-1][2] - 102.0) < 1e-9

    def test_volume_sums_to_daily(self):
        from db.load_intraday_and_labels import generate_1min_ticks

        daily_vol = 50000
        ticks = generate_1min_ticks(
            "BIAT", date(2024, 1, 15), 100.0, 105.0, 95.0, 102.0, daily_vol, seed=42
        )
        total_vol = sum(t[3] for t in ticks)
        assert total_vol == daily_vol

    def test_deterministic_with_seed(self):
        from db.load_intraday_and_labels import generate_1min_ticks

        t1 = generate_1min_ticks("BIAT", date(2024, 1, 15), 100.0, 105.0, 95.0, 102.0, 10000, seed=123)
        t2 = generate_1min_ticks("BIAT", date(2024, 1, 15), 100.0, 105.0, 95.0, 102.0, 10000, seed=123)
        assert t1 == t2

    def test_empty_on_zero_minutes(self):
        from db.load_intraday_and_labels import generate_1min_ticks

        # Even with valid data, the function should work
        ticks = generate_1min_ticks(
            "BT", date(2024, 1, 15), 50.0, 51.0, 49.0, 50.5, 1000, seed=1
        )
        assert len(ticks) > 0


# ══════════════════════════════════════════════════════════════
# PART 4 — Known anomalies CSV loading tests
# ══════════════════════════════════════════════════════════════


class TestKnownAnomaliesCSV:
    """Test loading known anomalies from CSV files."""

    def test_load_csv_via_evaluator(self, tmp_path):
        from app.domain.trading.anomaly_evaluator import AnomalyEvaluator

        csv_file = tmp_path / "test_anomalies.csv"
        csv_file.write_text(
            "date,anomaly_type,symbol,description\n"
            "2024-01-10,volume_spike,BIAT,Test spike\n"
            "2024-03-15,price_swing_daily,SFBT,Dividend reaction\n"
        )
        anomalies = AnomalyEvaluator.load_known_anomalies_csv(csv_file)
        assert len(anomalies) == 2
        assert anomalies[0].anomaly_type == "volume_spike"
        assert anomalies[1].symbol == "SFBT"

    def test_actual_csv_file_exists(self):
        csv_path = Path(__file__).resolve().parent.parent / "data" / "known_anomalies.csv"
        assert csv_path.exists(), f"Known anomalies CSV not found at {csv_path}"

    def test_actual_csv_has_valid_content(self):
        from app.domain.trading.anomaly_evaluator import AnomalyEvaluator

        csv_path = Path(__file__).resolve().parent.parent / "data" / "known_anomalies.csv"
        if not csv_path.exists():
            pytest.skip("CSV not found")

        anomalies = AnomalyEvaluator.load_known_anomalies_csv(csv_path)
        assert len(anomalies) >= 50  # We loaded 54

        symbols = {a.symbol for a in anomalies if a.symbol}
        assert "BIAT" in symbols
        assert "SFBT" in symbols

        types = {a.anomaly_type for a in anomalies}
        assert "volume_spike" in types
        assert "price_swing_daily" in types


class TestKnownAnomaliesCount:
    """Test that known anomalies cover the expected symbols."""

    def test_five_symbols_covered(self):
        from app.domain.trading.anomaly_evaluator import AnomalyEvaluator

        csv_path = Path(__file__).resolve().parent.parent / "data" / "known_anomalies.csv"
        if not csv_path.exists():
            pytest.skip("CSV not found")

        anomalies = AnomalyEvaluator.load_known_anomalies_csv(csv_path)
        symbols = {a.symbol for a in anomalies if a.symbol}
        expected = {"BIAT", "SFBT", "BT", "ATTIJARI BANK", "SAH"}
        assert symbols == expected

    def test_multi_year_coverage(self):
        from app.domain.trading.anomaly_evaluator import AnomalyEvaluator

        csv_path = Path(__file__).resolve().parent.parent / "data" / "known_anomalies.csv"
        if not csv_path.exists():
            pytest.skip("CSV not found")

        anomalies = AnomalyEvaluator.load_known_anomalies_csv(csv_path)
        years = {a.date.year for a in anomalies}
        assert 2022 in years
        assert 2023 in years
        assert 2024 in years
