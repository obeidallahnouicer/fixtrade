"""
Use case: Detect intraday anomalies.

Fetches intraday tick data from the repository and runs the
IntradayAnomalyService domain service on each trading day.
"""

from app.application.trading.dtos import AnomalyResult, DetectIntradayAnomaliesCommand
from app.domain.trading.intraday_anomaly_service import IntradayAnomalyService
from app.domain.trading.ports import IntradayTickRepository


class DetectIntradayAnomaliesUseCase:
    """Orchestrates intraday anomaly detection."""

    def __init__(
        self,
        tick_repo: IntradayTickRepository,
        service: IntradayAnomalyService | None = None,
    ) -> None:
        self._tick_repo = tick_repo
        self._service = service or IntradayAnomalyService()

    def execute(self, command: DetectIntradayAnomaliesCommand) -> list[AnomalyResult]:
        """Run intraday anomaly detection on recent tick data.

        Args:
            command: Contains symbol and days_back.

        Returns:
            List of AnomalyResult DTOs for detected anomalies.
        """
        from datetime import date, datetime, timedelta, time

        end_date = date.today()
        start_date = end_date - timedelta(days=command.days_back)

        ticks = self._tick_repo.get_ticks(
            symbol=command.symbol,
            start=datetime.combine(start_date, time.min),
            end=datetime.combine(end_date, time.max),
            tick_type="1min",
        )

        if not ticks:
            return []

        # Group ticks by date and detect per day
        from collections import defaultdict

        by_date: dict[date, list] = defaultdict(list)
        for t in ticks:
            by_date[t.timestamp.date()].append(t)

        all_results: list[AnomalyResult] = []

        for day_date in sorted(by_date):
            day_ticks = sorted(by_date[day_date], key=lambda t: t.timestamp)
            alerts = self._service.detect_intraday_anomalies(command.symbol, day_ticks)

            for alert in alerts:
                all_results.append(
                    AnomalyResult(
                        id=alert.id,
                        symbol=alert.symbol,
                        detected_at=alert.detected_at,
                        anomaly_type=alert.anomaly_type,
                        severity=alert.severity,
                        description=alert.description,
                    )
                )

        return all_results
