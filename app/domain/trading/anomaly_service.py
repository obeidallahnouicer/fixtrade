"""
Domain service: Anomaly detection logic.

Pure business logic for detecting market anomalies.
No framework imports. No IO. No side effects.

Detects:
    - Volume spikes (>3 standard deviations from mean)
    - Price swings (>5% change in recent period)
    - Suspicious patterns (unusual trading sequences)
    - Prediction contradictions (anomalies vs predictions)
    - Sentiment contradictions (market behavior vs news sentiment)
"""

from datetime import datetime
from decimal import Decimal
from statistics import mean, stdev
from typing import Optional
from uuid import uuid4

from app.domain.trading.entities import (
    AnomalyAlert,
    PricePrediction,
    SentimentScore,
    StockPrice,
)


class AnomalyDetectionService:
    """Domain service for detecting market anomalies.

    This is pure business logic. It takes in historical market data
    and returns detected anomalies. No database, no HTTP, no frameworks.
    """

    def __init__(
        self,
        volume_threshold_std: float = 3.0,
        price_change_threshold: Decimal = Decimal("0.05"),
        min_data_points: int = 20,
    ) -> None:
        """Initialize the anomaly detection service.

        Args:
            volume_threshold_std: Number of standard deviations for volume spikes.
            price_change_threshold: Minimum price change ratio to flag (e.g., 0.05 = 5%).
            min_data_points: Minimum historical data points required for analysis.
        """
        self._volume_threshold_std = volume_threshold_std
        self._price_change_threshold = price_change_threshold
        self._min_data_points = min_data_points

    def detect_anomalies(
        self,
        symbol: str,
        recent_prices: list[StockPrice],
        predictions: Optional[list[PricePrediction]] = None,
        sentiment_scores: Optional[list[SentimentScore]] = None,
    ) -> list[AnomalyAlert]:
        """Detect anomalies in recent market data.

        Args:
            symbol: Stock ticker symbol.
            recent_prices: List of recent OHLCV records (must be sorted by date ascending).
            predictions: Optional list of price predictions for cross-validation.
            sentiment_scores: Optional list of sentiment scores for cross-validation.

        Returns:
            List of detected anomaly alerts.
        """
        if len(recent_prices) < self._min_data_points:
            return []

        alerts = []

        # Check for volume spikes
        volume_alerts = self._detect_volume_spikes(symbol, recent_prices)
        alerts.extend(volume_alerts)

        # Check for price swings
        price_alerts = self._detect_price_swings(symbol, recent_prices)
        alerts.extend(price_alerts)

        # Check for suspicious patterns
        pattern_alerts = self._detect_suspicious_patterns(symbol, recent_prices)
        alerts.extend(pattern_alerts)

        # Check for prediction contradictions (if predictions provided)
        if predictions:
            prediction_alerts = self._detect_prediction_contradictions(
                symbol, recent_prices, predictions
            )
            alerts.extend(prediction_alerts)

        # Check for sentiment contradictions (if sentiment provided)
        if sentiment_scores:
            sentiment_alerts = self._detect_sentiment_contradictions(
                symbol, recent_prices, sentiment_scores
            )
            alerts.extend(sentiment_alerts)

        return alerts

    def _detect_volume_spikes(
        self, symbol: str, prices: list[StockPrice]
    ) -> list[AnomalyAlert]:
        """Detect volume spikes (>3 standard deviations from mean).

        Args:
            symbol: Stock ticker.
            prices: Historical price data.

        Returns:
            List of volume spike alerts.
        """
        if len(prices) < self._min_data_points:
            return []

        volumes = [p.volume for p in prices]
        avg_volume = mean(volumes)
        std_volume = stdev(volumes)

        if std_volume == 0:
            return []

        alerts = []
        latest_price = prices[-1]
        latest_volume = latest_price.volume

        # Calculate z-score
        z_score = (latest_volume - avg_volume) / std_volume

        if z_score > self._volume_threshold_std:
            severity = min(Decimal("1.0"), Decimal(str(z_score / 10)))
            description = (
                f"Volume spike detected: {latest_volume:,} "
                f"({z_score:.2f} std devs above mean of {avg_volume:,.0f})"
            )

            alerts.append(
                AnomalyAlert(
                    id=uuid4(),
                    symbol=symbol,
                    detected_at=datetime.now(),
                    anomaly_type="volume_spike",
                    severity=severity,
                    description=description,
                )
            )

        return alerts

    def _detect_price_swings(
        self, symbol: str, prices: list[StockPrice]
    ) -> list[AnomalyAlert]:
        """Detect abnormal price changes (>5% in recent period).

        Args:
            symbol: Stock ticker.
            prices: Historical price data.

        Returns:
            List of price swing alerts.
        """
        if len(prices) < 2:
            return []

        alerts = []

        # Check intraday swing (high to low)
        latest = prices[-1]
        if latest.high > 0 and latest.low > 0:
            intraday_change = (latest.high - latest.low) / latest.low
            if intraday_change > self._price_change_threshold:
                severity = min(Decimal("1.0"), intraday_change)
                description = (
                    f"Large intraday swing: {intraday_change*100:.2f}% "
                    f"(Low: {latest.low}, High: {latest.high})"
                )

                alerts.append(
                    AnomalyAlert(
                        id=uuid4(),
                        symbol=symbol,
                        detected_at=datetime.now(),
                        anomaly_type="price_swing_intraday",
                        severity=severity,
                        description=description,
                    )
                )

        # Check day-over-day change
        if len(prices) >= 2:
            previous = prices[-2]
            if previous.close > 0:
                daily_change = abs(latest.close - previous.close) / previous.close
                if daily_change > self._price_change_threshold:
                    severity = min(Decimal("1.0"), daily_change)
                    direction = "increase" if latest.close > previous.close else "decrease"
                    description = (
                        f"Sharp price {direction}: {daily_change*100:.2f}% "
                        f"({previous.close} → {latest.close})"
                    )

                    alerts.append(
                        AnomalyAlert(
                            id=uuid4(),
                            symbol=symbol,
                            detected_at=datetime.now(),
                            anomaly_type="price_swing_daily",
                            severity=severity,
                            description=description,
                        )
                    )

        return alerts

    def _detect_suspicious_patterns(
        self, symbol: str, prices: list[StockPrice]
    ) -> list[AnomalyAlert]:
        """Detect suspicious trading patterns.

        Checks for:
            - Consecutive days of zero volume
            - Unusual price stability (no movement for extended period)

        Args:
            symbol: Stock ticker.
            prices: Historical price data.

        Returns:
            List of pattern alerts.
        """
        if len(prices) < 5:
            return []

        alerts = []

        # Check for consecutive zero volume days
        recent = prices[-5:]
        zero_volume_count = sum(1 for p in recent if p.volume == 0)
        if zero_volume_count >= 3:
            description = (
                f"Suspicious pattern: {zero_volume_count}/5 recent days with zero volume"
            )
            alerts.append(
                AnomalyAlert(
                    id=uuid4(),
                    symbol=symbol,
                    detected_at=datetime.now(),
                    anomaly_type="zero_volume_pattern",
                    severity=Decimal("0.6"),
                    description=description,
                )
            )

        # Check for unusual price stability (all closes identical)
        unique_closes = len({p.close for p in recent})
        if unique_closes == 1 and recent[0].close > 0:
            description = f"Suspicious pattern: Price unchanged for 5 consecutive days at {recent[0].close}"
            alerts.append(
                AnomalyAlert(
                    id=uuid4(),
                    symbol=symbol,
                    detected_at=datetime.now(),
                    anomaly_type="price_stagnation",
                    severity=Decimal("0.4"),
                    description=description,
                )
            )

        return alerts

    def _detect_prediction_contradictions(
        self,
        symbol: str,
        prices: list[StockPrice],
        predictions: list[PricePrediction],
    ) -> list[AnomalyAlert]:
        """Detect contradictions between market behavior and predictions.

        Flags situations like:
            - Volume spike but price predicted to drop (pump and dump?)
            - Price surging but predicted to fall (manipulation?)
            - Low volume but price matching bullish prediction (thin market risk)

        Args:
            symbol: Stock ticker.
            prices: Historical price data.
            predictions: Price predictions for upcoming days.

        Returns:
            List of contradiction alerts.
        """
        if len(prices) < 2 or not predictions:
            return []

        alerts = []
        latest = prices[-1]
        previous = prices[-2]

        # Calculate recent volume trend
        recent_volumes = [p.volume for p in prices[-5:]]
        avg_volume = mean(recent_volumes)
        volume_spike = latest.volume > avg_volume * 2 if avg_volume > 0 else False

        # Calculate recent price movement
        price_change_pct = (
            (latest.close - previous.close) / previous.close if previous.close > 0 else Decimal("0")
        )

        # Get nearest prediction (first one, usually tomorrow)
        nearest_prediction = predictions[0]
        predicted_change_pct = (
            (nearest_prediction.predicted_close - latest.close) / latest.close
            if latest.close > 0
            else Decimal("0")
        )

        # CONTRADICTION 1: Volume spike + bearish prediction
        if volume_spike and predicted_change_pct < Decimal("-0.03"):
            severity = min(Decimal("1.0"), abs(predicted_change_pct))
            description = (
                f"⚠️ Contradiction: Volume spike ({latest.volume:,}) detected "
                f"but price predicted to drop {predicted_change_pct*100:.1f}% "
                f"(from {latest.close} to {nearest_prediction.predicted_close}). "
                f"Possible pump-and-dump or manipulation."
            )
            alerts.append(
                AnomalyAlert(
                    id=uuid4(),
                    symbol=symbol,
                    detected_at=datetime.now(),
                    anomaly_type="prediction_contradiction_bearish",
                    severity=severity,
                    description=description,
                )
            )

        # CONTRADICTION 2: Price rising sharply + bearish prediction
        if price_change_pct > Decimal("0.05") and predicted_change_pct < Decimal("-0.03"):
            severity = Decimal("0.8")
            description = (
                f"⚠️ Contradiction: Price surged {price_change_pct*100:.1f}% today "
                f"but predicted to drop {predicted_change_pct*100:.1f}% tomorrow. "
                f"Current: {latest.close}, Predicted: {nearest_prediction.predicted_close}. "
                f"High reversal risk or data anomaly."
            )
            alerts.append(
                AnomalyAlert(
                    id=uuid4(),
                    symbol=symbol,
                    detected_at=datetime.now(),
                    anomaly_type="prediction_contradiction_reversal",
                    severity=severity,
                    description=description,
                )
            )

        # CONTRADICTION 3: Low volume but strong bullish prediction
        if latest.volume < avg_volume * 0.5 and predicted_change_pct > Decimal("0.05"):
            severity = Decimal("0.6")
            description = (
                f"⚠️ Warning: Very low volume ({latest.volume:,}, {avg_volume * 0.5:.0f} below average) "
                f"but price predicted to rise {predicted_change_pct*100:.1f}%. "
                f"Thin market risk - prediction may not materialize without volume support."
            )
            alerts.append(
                AnomalyAlert(
                    id=uuid4(),
                    symbol=symbol,
                    detected_at=datetime.now(),
                    anomaly_type="prediction_low_volume_risk",
                    severity=severity,
                    description=description,
                )
            )

        # CONTRADICTION 4: Volume spike + bullish prediction (confirmation, but flag for monitoring)
        if volume_spike and predicted_change_pct > Decimal("0.05"):
            severity = Decimal("0.3")  # Low severity, this is actually good alignment
            description = (
                f"ℹ️ Note: Strong volume ({latest.volume:,}) aligns with bullish prediction "
                f"(+{predicted_change_pct*100:.1f}%). Monitor for sustained momentum."
            )
            alerts.append(
                AnomalyAlert(
                    id=uuid4(),
                    symbol=symbol,
                    detected_at=datetime.now(),
                    anomaly_type="prediction_bullish_confirmation",
                    severity=severity,
                    description=description,
                )
            )

        return alerts

    def _detect_sentiment_contradictions(
        self,
        symbol: str,
        prices: list[StockPrice],
        sentiment_scores: list[SentimentScore],
    ) -> list[AnomalyAlert]:
        """Detect contradictions between market behavior and news sentiment.

        Flags situations like:
            - Negative sentiment but price spike (manipulation?)
            - Positive sentiment but price crash (hidden problems?)
            - Volume spike with negative news (panic selling or buying opportunity?)

        CRITICAL: Uses date matching to ensure sentiment aligns with price data.

        Args:
            symbol: Stock ticker.
            prices: Historical price data.
            sentiment_scores: Sentiment scores for recent dates.

        Returns:
            List of sentiment contradiction alerts.
        """
        if len(prices) < 2 or not sentiment_scores:
            return []

        alerts = []
        latest = prices[-1]
        previous = prices[-2]

        # Find sentiment for the latest date (DATE MATCHING IS CRITICAL!)
        latest_sentiment = None
        for sentiment in sentiment_scores:
            if sentiment.date == latest.date:
                latest_sentiment = sentiment
                break

        # If no sentiment for today, try yesterday
        if not latest_sentiment:
            for sentiment in sentiment_scores:
                if sentiment.date == previous.date:
                    latest_sentiment = sentiment
                    break

        if not latest_sentiment:
            # No sentiment data available for recent dates
            return alerts

        # Calculate price movement
        price_change_pct = (
            (latest.close - previous.close) / previous.close 
            if previous.close > 0 
            else Decimal("0")
        )

        # Calculate volume anomaly
        recent_volumes = [p.volume for p in prices[-5:]]
        avg_volume = mean(recent_volumes)
        volume_spike = latest.volume > avg_volume * 2 if avg_volume > 0 else False

        # Extract sentiment info
        sentiment_label = latest_sentiment.sentiment.lower()
        sentiment_score = latest_sentiment.score

        # CONTRADICTION 1: Negative sentiment but price spike
        if sentiment_label == "negative" and price_change_pct > Decimal("0.03"):
            severity = Decimal("0.9")
            description = (
                f"🚨 MANIPULATION ALERT: Negative news sentiment "
                f"(score: {sentiment_score}, {latest_sentiment.article_count} articles on {latest_sentiment.date}) "
                f"but price surged +{price_change_pct*100:.1f}% "
                f"({previous.close} → {latest.close}). "
                f"Possible pump scheme or hidden positive catalyst."
            )
            alerts.append(
                AnomalyAlert(
                    id=uuid4(),
                    symbol=symbol,
                    detected_at=datetime.now(),
                    anomaly_type="sentiment_contradiction_bullish",
                    severity=severity,
                    description=description,
                )
            )

        # CONTRADICTION 2: Positive sentiment but price crash
        if sentiment_label == "positive" and price_change_pct < Decimal("-0.03"):
            severity = Decimal("0.9")
            description = (
                f"🚨 WARNING: Positive news sentiment "
                f"(score: {sentiment_score}, {latest_sentiment.article_count} articles on {latest_sentiment.date}) "
                f"but price crashed {price_change_pct*100:.1f}% "
                f"({previous.close} → {latest.close}). "
                f"Hidden problems, profit-taking, or market-wide pressure."
            )
            alerts.append(
                AnomalyAlert(
                    id=uuid4(),
                    symbol=symbol,
                    detected_at=datetime.now(),
                    anomaly_type="sentiment_contradiction_bearish",
                    severity=severity,
                    description=description,
                )
            )

        # CONTRADICTION 3: Negative sentiment + volume spike
        if sentiment_label == "negative" and volume_spike:
            severity = Decimal("0.7")
            description = (
                f"⚠️ Panic signal: Negative sentiment "
                f"(score: {sentiment_score} on {latest_sentiment.date}) + volume spike "
                f"({latest.volume:,} vs avg {avg_volume:,.0f}). "
                f"Could indicate panic selling OR contrarian buying opportunity."
            )
            alerts.append(
                AnomalyAlert(
                    id=uuid4(),
                    symbol=symbol,
                    detected_at=datetime.now(),
                    anomaly_type="sentiment_negative_volume_spike",
                    severity=severity,
                    description=description,
                )
            )

        # CONTRADICTION 4: Strong positive sentiment but no price reaction
        if (
            sentiment_label == "positive" 
            and sentiment_score > Decimal("0.5")
            and abs(price_change_pct) < Decimal("0.01")
        ):
            severity = Decimal("0.4")
            description = (
                f"ℹ️ Note: Strong positive sentiment "
                f"(score: {sentiment_score}, {latest_sentiment.article_count} articles) "
                f"but price unchanged ({latest.close}). "
                f"Market may not believe the news or waiting for confirmation."
            )
            alerts.append(
                AnomalyAlert(
                    id=uuid4(),
                    symbol=symbol,
                    detected_at=datetime.now(),
                    anomaly_type="sentiment_ignored",
                    severity=severity,
                    description=description,
                )
            )

        # CONTRADICTION 5: Strong negative sentiment but price stable/rising (contrarian signal)
        if (
            sentiment_label == "negative"
            and sentiment_score < Decimal("-0.5")
            and price_change_pct >= Decimal("0")
        ):
            severity = Decimal("0.5")
            description = (
                f"💡 Contrarian signal: Strong negative sentiment "
                f"(score: {sentiment_score}) but price holding/rising "
                f"({price_change_pct*100:+.1f}%). "
                f"Market may have priced in bad news or sees value."
            )
            alerts.append(
                AnomalyAlert(
                    id=uuid4(),
                    symbol=symbol,
                    detected_at=datetime.now(),
                    anomaly_type="sentiment_contrarian_bullish",
                    severity=severity,
                    description=description,
                )
            )

        return alerts
