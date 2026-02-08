"""
Anomaly detection evaluation framework.

Compares detected anomalies against labeled ground-truth data to
compute Precision, Recall, and F1-Score. Supports:

    1. Backtesting the AnomalyDetectionService on historical data
    2. Loading known anomalies from CSV/JSON for validation
    3. Per-anomaly-type breakdown
    4. Configurable date tolerance (±N days for matching)

Usage:
    evaluator = AnomalyEvaluator()
    report = evaluator.backtest(
        symbol="BIAT",
        prices=historical_prices,
        known_anomalies=labeled_anomalies,
    )
    print(report.summary())
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pandas as pd

from app.domain.trading.anomaly_service import AnomalyDetectionService
from app.domain.trading.entities import (
    AnomalyAlert,
    PricePrediction,
    SentimentScore,
    StockPrice,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LabeledAnomaly:
    """A known/confirmed anomaly from the validation dataset.

    Attributes:
        date: Date the anomaly occurred.
        anomaly_type: Category (volume_spike, price_swing, etc.).
                      Use "any" to match any detected type.
        symbol: Stock ticker (optional, defaults to the backtest symbol).
        description: Human-readable note about the anomaly.
    """

    date: date
    anomaly_type: str = "any"
    symbol: str | None = None
    description: str = ""


@dataclass
class EvaluationMetrics:
    """Precision / Recall / F1-Score for anomaly detection."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def support(self) -> int:
        """Total ground-truth positives."""
        return self.true_positives + self.false_negatives


@dataclass
class EvaluationReport:
    """Full evaluation report with overall and per-type metrics."""

    symbol: str
    overall: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    per_type: dict[str, EvaluationMetrics] = field(default_factory=dict)
    total_detected: int = 0
    total_known: int = 0
    matched_pairs: list[tuple[LabeledAnomaly, AnomalyAlert]] = field(
        default_factory=list,
    )
    unmatched_known: list[LabeledAnomaly] = field(default_factory=list)
    unmatched_detected: list[AnomalyAlert] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"╔══════════════════════════════════════════════════════════╗",
            f"║  Anomaly Detection Evaluation — {self.symbol:>10s}           ║",
            f"╠══════════════════════════════════════════════════════════╣",
            f"║  Detected: {self.total_detected:>4d}   Known: {self.total_known:>4d}                       ║",
            f"║  TP: {self.overall.true_positives:>4d}   FP: {self.overall.false_positives:>4d}   FN: {self.overall.false_negatives:>4d}                 ║",
            f"╠══════════════════════════════════════════════════════════╣",
            f"║  Precision : {self.overall.precision:>6.2%}                                ║",
            f"║  Recall    : {self.overall.recall:>6.2%}                                ║",
            f"║  F1-Score  : {self.overall.f1_score:>6.2%}                                ║",
            f"╠══════════════════════════════════════════════════════════╣",
        ]

        if self.per_type:
            lines.append(
                f"║  {'Type':<28s} {'P':>6s} {'R':>6s} {'F1':>6s} {'S':>4s} ║"
            )
            lines.append(
                f"║  {'─'*28} {'─'*6} {'─'*6} {'─'*6} {'─'*4} ║"
            )
            for atype, m in sorted(self.per_type.items()):
                lines.append(
                    f"║  {atype:<28s} {m.precision:>5.1%} {m.recall:>5.1%} "
                    f"{m.f1_score:>5.1%} {m.support:>4d} ║"
                )

        lines.append(
            f"╚══════════════════════════════════════════════════════════╝"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "symbol": self.symbol,
            "total_detected": self.total_detected,
            "total_known": self.total_known,
            "overall": {
                "precision": round(self.overall.precision, 4),
                "recall": round(self.overall.recall, 4),
                "f1_score": round(self.overall.f1_score, 4),
                "true_positives": self.overall.true_positives,
                "false_positives": self.overall.false_positives,
                "false_negatives": self.overall.false_negatives,
            },
            "per_type": {
                atype: {
                    "precision": round(m.precision, 4),
                    "recall": round(m.recall, 4),
                    "f1_score": round(m.f1_score, 4),
                    "support": m.support,
                }
                for atype, m in self.per_type.items()
            },
        }


# ══════════════════════════════════════════════════════════════════════
# Evaluator
# ══════════════════════════════════════════════════════════════════════


class AnomalyEvaluator:
    """Evaluates the AnomalyDetectionService against known anomalies.

    Args:
        service: Optional pre-configured AnomalyDetectionService.
        date_tolerance_days: Days of tolerance for matching (±N days).
    """

    def __init__(
        self,
        service: AnomalyDetectionService | None = None,
        date_tolerance_days: int = 1,
    ) -> None:
        self._service = service or AnomalyDetectionService()
        self._tolerance = timedelta(days=date_tolerance_days)

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def backtest(
        self,
        symbol: str,
        prices: list[StockPrice],
        known_anomalies: list[LabeledAnomaly],
        predictions: list[PricePrediction] | None = None,
        sentiment_scores: list[SentimentScore] | None = None,
    ) -> EvaluationReport:
        """Run the detection service on historical data and evaluate.

        Walks through the price history with a sliding window,
        detecting anomalies at each step. Then matches detected
        anomalies against the known list.

        Args:
            symbol: Stock ticker.
            prices: Full historical price series (sorted by date asc).
            known_anomalies: Labeled anomalies for validation.
            predictions: Optional price predictions for cross-validation.
            sentiment_scores: Optional sentiment scores.

        Returns:
            An EvaluationReport with Precision/Recall/F1.
        """
        if not prices:
            return EvaluationReport(symbol=symbol)

        # Run detection on the full series
        detected = self._service.detect_anomalies(
            symbol=symbol,
            recent_prices=prices,
            predictions=predictions,
            sentiment_scores=sentiment_scores,
        )

        # Also run sliding-window detection for anomalies at different points
        window_size = max(self._service._min_data_points, 20)
        all_detected: list[AnomalyAlert] = list(detected)
        seen_dates: set[tuple[str, date]] = set()

        # De-dup from full-series run
        for a in all_detected:
            seen_dates.add((a.anomaly_type, a.detected_at.date()))

        # Sliding window: step through data to detect anomalies at each point
        for end_idx in range(window_size, len(prices)):
            window = prices[end_idx - window_size : end_idx + 1]
            window_alerts = self._service.detect_anomalies(
                symbol=symbol,
                recent_prices=window,
                predictions=predictions,
                sentiment_scores=sentiment_scores,
            )
            for alert in window_alerts:
                key = (alert.anomaly_type, window[-1].date)
                if key not in seen_dates:
                    seen_dates.add(key)
                    all_detected.append(alert)

        logger.info(
            "Backtest for %s: %d anomalies detected, %d known",
            symbol, len(all_detected), len(known_anomalies),
        )

        return self._evaluate(symbol, all_detected, known_anomalies)

    # ------------------------------------------------------------------
    # Matching & Evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        symbol: str,
        detected: list[AnomalyAlert],
        known: list[LabeledAnomaly],
    ) -> EvaluationReport:
        """Match detected vs known anomalies and compute metrics."""
        report = EvaluationReport(
            symbol=symbol,
            total_detected=len(detected),
            total_known=len(known),
        )

        # Track which known anomalies have been matched
        known_matched: set[int] = set()
        detected_matched: set[int] = set()

        for d_idx, det in enumerate(detected):
            det_date = (
                det.detected_at.date()
                if hasattr(det.detected_at, "date")
                else det.detected_at
            )
            matched = False

            for k_idx, lab in enumerate(known):
                if k_idx in known_matched:
                    continue

                # Date tolerance check
                if abs((det_date - lab.date).days) > self._tolerance.days:
                    continue

                # Type matching: "any" matches everything
                if lab.anomaly_type != "any" and lab.anomaly_type != det.anomaly_type:
                    continue

                # Match found!
                report.matched_pairs.append((lab, det))
                known_matched.add(k_idx)
                detected_matched.add(d_idx)
                matched = True
                break

            if not matched:
                report.unmatched_detected.append(det)

        # Known anomalies that weren't detected
        for k_idx, lab in enumerate(known):
            if k_idx not in known_matched:
                report.unmatched_known.append(lab)

        # Compute overall metrics
        report.overall.true_positives = len(report.matched_pairs)
        report.overall.false_positives = len(report.unmatched_detected)
        report.overall.false_negatives = len(report.unmatched_known)

        # Compute per-type metrics
        self._compute_per_type_metrics(report, detected, known)

        return report

    def _compute_per_type_metrics(
        self,
        report: EvaluationReport,
        detected: list[AnomalyAlert],
        known: list[LabeledAnomaly],
    ) -> None:
        """Compute Precision/Recall/F1 broken down by anomaly type."""
        # Gather all unique types
        types_from_detected = {a.anomaly_type for a in detected}
        types_from_known = {
            a.anomaly_type for a in known if a.anomaly_type != "any"
        }
        all_types = types_from_detected | types_from_known

        for atype in sorted(all_types):
            m = EvaluationMetrics()

            # Count TP: matched pairs where detected type == atype
            for lab, det in report.matched_pairs:
                if det.anomaly_type == atype:
                    m.true_positives += 1

            # Count FP: unmatched detections of this type
            for det in report.unmatched_detected:
                if det.anomaly_type == atype:
                    m.false_positives += 1

            # Count FN: unmatched known of this type (or "any")
            for lab in report.unmatched_known:
                if lab.anomaly_type == atype or lab.anomaly_type == "any":
                    m.false_negatives += 1

            report.per_type[atype] = m

    # ------------------------------------------------------------------
    # Load known anomalies from files
    # ------------------------------------------------------------------

    @staticmethod
    def load_known_anomalies_csv(path: str | Path) -> list[LabeledAnomaly]:
        """Load labeled anomalies from a CSV file.

        Expected columns: date, anomaly_type, symbol (optional), description (optional)

        Args:
            path: Path to the CSV file.

        Returns:
            List of LabeledAnomaly objects.
        """
        df = pd.read_csv(path, comment="#")
        anomalies = []
        for _, row in df.iterrows():
            anomalies.append(
                LabeledAnomaly(
                    date=pd.to_datetime(row["date"]).date(),
                    anomaly_type=str(row.get("anomaly_type", "any")),
                    symbol=str(row["symbol"]) if "symbol" in row and pd.notna(row.get("symbol")) else None,
                    description=str(row.get("description", "")),
                )
            )
        return anomalies

    @staticmethod
    def load_known_anomalies_json(path: str | Path) -> list[LabeledAnomaly]:
        """Load labeled anomalies from a JSON file.

        Expected format: list of objects with keys:
            date (YYYY-MM-DD), anomaly_type, symbol, description

        Args:
            path: Path to the JSON file.

        Returns:
            List of LabeledAnomaly objects.
        """
        with open(path) as f:
            data = json.load(f)

        return [
            LabeledAnomaly(
                date=date.fromisoformat(item["date"]),
                anomaly_type=item.get("anomaly_type", "any"),
                symbol=item.get("symbol"),
                description=item.get("description", ""),
            )
            for item in data
        ]

    # ------------------------------------------------------------------
    # Generate synthetic known anomalies from historical data
    # ------------------------------------------------------------------

    @staticmethod
    def generate_known_anomalies_from_data(
        prices: list[StockPrice],
        volume_z_threshold: float = 3.0,
        price_change_threshold: float = 0.05,
    ) -> list[LabeledAnomaly]:
        """Auto-generate ground-truth anomalies from raw price data.

        Uses simple statistical rules to label obvious anomalies.
        Useful when no hand-labeled dataset is available.

        Args:
            prices: Historical OHLCV data sorted by date.
            volume_z_threshold: Z-score threshold for volume spikes.
            price_change_threshold: Fraction for price swings (e.g. 0.05 = 5%).

        Returns:
            List of auto-generated LabeledAnomaly objects.
        """
        if len(prices) < 20:
            return []

        from statistics import mean, stdev

        anomalies: list[LabeledAnomaly] = []

        volumes = [p.volume for p in prices]
        avg_vol = mean(volumes)
        std_vol = stdev(volumes) if len(volumes) > 1 else 0

        for i in range(1, len(prices)):
            p = prices[i]
            prev = prices[i - 1]

            # Volume spike
            if std_vol > 0:
                z = (p.volume - avg_vol) / std_vol
                if z > volume_z_threshold:
                    anomalies.append(
                        LabeledAnomaly(
                            date=p.date,
                            anomaly_type="volume_spike",
                            symbol=p.symbol,
                            description=f"Volume Z-score={z:.2f}",
                        )
                    )

            # Daily price swing
            if prev.close > 0:
                change = abs(float(p.close - prev.close) / float(prev.close))
                if change > price_change_threshold:
                    anomalies.append(
                        LabeledAnomaly(
                            date=p.date,
                            anomaly_type="price_swing_daily",
                            symbol=p.symbol,
                            description=f"Price change={change:.2%}",
                        )
                    )

            # Intraday swing
            if p.low > 0:
                intraday = float(p.high - p.low) / float(p.low)
                if intraday > price_change_threshold:
                    anomalies.append(
                        LabeledAnomaly(
                            date=p.date,
                            anomaly_type="price_swing_intraday",
                            symbol=p.symbol,
                            description=f"Intraday swing={intraday:.2%}",
                        )
                    )

            # Zero volume
            if p.volume == 0:
                anomalies.append(
                    LabeledAnomaly(
                        date=p.date,
                        anomaly_type="zero_volume_pattern",
                        symbol=p.symbol,
                        description="Zero volume day",
                    )
                )

        return anomalies
