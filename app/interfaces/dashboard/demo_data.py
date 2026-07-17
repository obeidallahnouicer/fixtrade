"""Small, deterministic fallback dataset for the interactive demo."""

from __future__ import annotations

import csv
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from app.interfaces.trading.schemas import (
    AnomalyItem,
    HistoricalPriceItem,
    PredictPriceItem,
    RecommendationResponse,
    SentimentResponse,
)

DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw"
FILES = ("histo_cotation_2025.csv", "histo_cotation_2024.csv")


def _number(value: str) -> Decimal:
    return Decimal(value.strip() or "0")


@lru_cache(maxsize=32)
def load_latest_history(symbol: str, limit: int = 45) -> tuple[HistoricalPriceItem, ...]:
    """Read the latest bundled BVMT closes without requiring PostgreSQL."""
    rows: list[HistoricalPriceItem] = []
    wanted = symbol.strip().upper()

    for filename in FILES:
        path = DATA_ROOT / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter=";")
            next(reader, None)
            for row in reader:
                if len(row) < 6:
                    continue
                name = row[3].strip().upper()
                if name != wanted:
                    continue
                try:
                    rows.append(
                        HistoricalPriceItem(
                            date=datetime.strptime(
                                row[0].strip(), "%d/%m/%Y"
                            ).date(),
                            close=_number(row[5]),
                        )
                    )
                except (ValueError, ArithmeticError):
                    continue
        if rows:
            break

    rows.sort(key=lambda item: item.date)
    return tuple(rows[-limit:])


def build_demo_predictions(
    symbol: str, history: list[HistoricalPriceItem], horizon: int = 5
) -> list[PredictPriceItem]:
    if not history:
        return []

    closes = [item.close for item in history[-6:]]
    last_close = closes[-1]
    trend = (
        (last_close - closes[0]) / Decimal(max(len(closes) - 1, 1))
        if len(closes) > 1
        else Decimal("0")
    )
    max_step = last_close * Decimal("0.015")
    step = max(-max_step, min(max_step, trend))
    spread = last_close * Decimal("0.025")

    predictions: list[PredictPriceItem] = []
    target = history[-1].date
    for day_index in range(1, horizon + 1):
        target += timedelta(days=1)
        while target.weekday() >= 5:
            target += timedelta(days=1)
        predicted = max(Decimal("0.001"), last_close + (step * day_index))
        predictions.append(
            PredictPriceItem(
                symbol=symbol,
                target_date=target,
                predicted_close=predicted.quantize(Decimal("0.001")),
                confidence_lower=max(
                    Decimal("0.001"), predicted - spread
                ).quantize(Decimal("0.001")),
                confidence_upper=(predicted + spread).quantize(Decimal("0.001")),
            )
        )
    return predictions


def build_demo_sentiment(
    symbol: str, history: list[HistoricalPriceItem]
) -> SentimentResponse | None:
    if len(history) < 2:
        return None
    first, last = history[-6].close if len(history) >= 6 else history[0].close, history[-1].close
    change = (last - first) / first if first else Decimal("0")
    score = max(Decimal("-0.75"), min(Decimal("0.75"), change * Decimal("4")))
    label = "positive" if score > Decimal("0.08") else "negative" if score < Decimal("-0.08") else "neutral"
    return SentimentResponse(
        symbol=symbol,
        date=history[-1].date,
        score=score.quantize(Decimal("0.01")),
        sentiment=label,
        article_count=12,
    )


def build_demo_recommendation(
    symbol: str,
    history: list[HistoricalPriceItem],
    predictions: list[PredictPriceItem],
) -> RecommendationResponse | None:
    if not history or not predictions:
        return None
    current = history[-1].close
    expected = (predictions[-1].predicted_close - current) / current if current else Decimal("0")
    action = "BUY" if expected > Decimal("0.015") else "SELL" if expected < Decimal("-0.015") else "HOLD"
    confidence = min(Decimal("0.86"), Decimal("0.58") + abs(expected) * Decimal("4"))
    return RecommendationResponse(
        symbol=symbol,
        action=action,
        confidence=confidence.quantize(Decimal("0.01")),
        reasoning=(
            f"Demo signal based on the latest bundled BVMT prices: "
            f"the five-session projection implies {expected * 100:.1f}% movement."
        ),
    )


def build_demo_anomalies(
    symbol: str, history: list[HistoricalPriceItem]
) -> list[AnomalyItem]:
    if len(history) < 2:
        return []
    previous, latest = history[-2], history[-1]
    move = abs((latest.close - previous.close) / previous.close) if previous.close else Decimal("0")
    if move < Decimal("0.025"):
        return []
    return [
        AnomalyItem(
            id=uuid5(NAMESPACE_URL, f"fixtrade-demo:{symbol}:{latest.date}"),
            symbol=symbol,
            detected_at=datetime.combine(
                latest.date, time(hour=16), tzinfo=timezone.utc
            ),
            anomaly_type="Price move",
            severity=min(Decimal("0.95"), move * Decimal("8")).quantize(
                Decimal("0.01")
            ),
            description=f"Latest close moved {move * 100:.1f}% from the prior session.",
        )
    ]
