"""
Adapter: Anomaly detection engine.

Implements AnomalyDetectionPort.
Responsible for running anomaly detection algorithms on market data.
Integrates with prediction models and sentiment analysis for cross-validation.
Optionally dispatches alerts through the notification system.
"""

import asyncio
import logging
from datetime import date, timedelta

from app.domain.trading.anomaly_notifier import AnomalyNotifier
from app.domain.trading.anomaly_service import AnomalyDetectionService
from app.domain.trading.entities import AnomalyAlert
from app.domain.trading.ports import (
    AnomalyAlertRepository,
    AnomalyDetectionPort,
    PricePredictionPort,
    SentimentAnalysisPort,
    StockPriceRepository,
)

logger = logging.getLogger(__name__)


class AnomalyDetectionAdapter(AnomalyDetectionPort):
    """Concrete adapter for market anomaly detection.

    Implements the AnomalyDetectionPort defined in the domain layer.
    Fetches market data, predictions, sentiment scores, runs anomaly detection
    with cross-validation, persists results, and optionally pushes notifications.
    """

    def __init__(
        self,
        price_repo: StockPriceRepository,
        alert_repo: AnomalyAlertRepository,
        prediction_port: PricePredictionPort | None = None,
        sentiment_port: SentimentAnalysisPort | None = None,
        notifier: AnomalyNotifier | None = None,
        lookback_days: int = 30,
        enable_prediction_check: bool = True,
        enable_sentiment_check: bool = True,
    ) -> None:
        """Initialize the anomaly detection adapter.

        Args:
            price_repo: Repository for fetching historical price data.
            alert_repo: Repository for persisting detected anomalies.
            prediction_port: Optional port for fetching price predictions.
            sentiment_port: Optional port for fetching sentiment scores.
            notifier: Optional AnomalyNotifier for alert dispatch.
            lookback_days: Number of days of historical data to analyze.
            enable_prediction_check: Whether to cross-validate with predictions.
            enable_sentiment_check: Whether to cross-validate with sentiment.
        """
        self._price_repo = price_repo
        self._alert_repo = alert_repo
        self._prediction_port = prediction_port
        self._sentiment_port = sentiment_port
        self._notifier = notifier
        self._lookback_days = lookback_days
        self._enable_prediction_check = enable_prediction_check
        self._enable_sentiment_check = enable_sentiment_check
        self._detection_service = AnomalyDetectionService()

    def detect(self, symbol: str) -> list[AnomalyAlert]:
        """Return list of detected anomalies for a symbol.

        Fetches recent market data, predictions, sentiment scores,
        runs anomaly detection with cross-validation, persists new alerts,
        and returns them.

        Args:
            symbol: BVMT stock ticker.

        Returns:
            List of AnomalyAlert entities.
        """
        # Fetch recent market data
        end_date = date.today()
        start_date = end_date - timedelta(days=self._lookback_days)

        recent_prices = self._price_repo.get_history(
            symbol=symbol, start=start_date, end=end_date
        )

        if not recent_prices:
            return []

        # Optionally fetch predictions for cross-validation
        predictions = None
        if self._enable_prediction_check and self._prediction_port:
            try:
                predictions = self._prediction_port.predict(
                    symbol=symbol, horizon_days=5
                )
            except Exception:
                # If prediction fails, continue without it
                predictions = None

        # Optionally fetch sentiment scores for cross-validation (DATE-AWARE!)
        sentiment_scores = []
        if self._enable_sentiment_check and self._sentiment_port:
            try:
                # Fetch sentiment for last few days to match with price data
                for days_ago in range(5):
                    target_date = end_date - timedelta(days=days_ago)
                    try:
                        sentiment = self._sentiment_port.get_sentiment(
                            symbol=symbol, target_date=target_date
                        )
                        sentiment_scores.append(sentiment)
                    except Exception:
                        # No sentiment for this date, continue
                        continue
            except Exception:
                # If sentiment fails, continue without it
                sentiment_scores = []

        # Run anomaly detection (with predictions and sentiment if available)
        alerts = self._detection_service.detect_anomalies(
            symbol=symbol,
            recent_prices=recent_prices,
            predictions=predictions,
            sentiment_scores=sentiment_scores if sentiment_scores else None,
        )

        # Persist detected anomalies
        if alerts:
            self._alert_repo.save_batch(alerts)

            # Dispatch notifications (fire-and-forget)
            if self._notifier is not None:
                self._dispatch_notifications(alerts)

        return alerts

    def _dispatch_notifications(self, alerts: list[AnomalyAlert]) -> None:
        """Send notifications for detected anomalies.

        Tries to use the running event loop; if none exists,
        creates a new one (e.g. when called from sync context).
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._notifier.notify(alerts))
        except RuntimeError:
            # No running loop — run synchronously
            try:
                asyncio.run(self._notifier.notify(alerts))
            except Exception:
                logger.warning(
                    "Could not dispatch anomaly notifications",
                    exc_info=True,
                )

