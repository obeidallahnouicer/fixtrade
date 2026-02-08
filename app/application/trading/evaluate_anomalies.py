"""
Use case: Evaluate anomaly detection performance.

Input: EvaluateAnomaliesCommand (symbol, days_back, date_tolerance_days)
Output: EvaluateAnomaliesResult with Precision / Recall / F1-Score
Side effects: None (read-only).
"""

import logging

from app.application.trading.dtos import (
    EvaluateAnomaliesCommand,
    EvaluateAnomaliesResult,
    EvaluationMetricsResult,
    PerTypeMetricsResult,
)
from app.domain.trading.anomaly_evaluator import AnomalyEvaluator
from app.domain.trading.ports import StockPriceRepository

logger = logging.getLogger(__name__)


class EvaluateAnomaliesUseCase:
    """Orchestrates anomaly detection backtesting and evaluation.

    Fetches historical data, generates ground-truth labels using
    statistical rules, runs the detection service on the same data,
    and compares the results to compute P/R/F1.
    """

    def __init__(
        self,
        price_repo: StockPriceRepository,
        evaluator: AnomalyEvaluator | None = None,
    ) -> None:
        self._price_repo = price_repo
        self._evaluator = evaluator or AnomalyEvaluator()

    def execute(self, command: EvaluateAnomaliesCommand) -> EvaluateAnomaliesResult:
        """Run the evaluation backtest.

        Args:
            command: Contains symbol, days_back, and date_tolerance_days.

        Returns:
            Evaluation result with overall and per-type P/R/F1 metrics.
        """
        from datetime import date, timedelta

        logger.info(
            "Evaluating anomaly detection for %s over %d days",
            command.symbol,
            command.days_back,
        )

        # Fetch historical data
        end_date = date.today()
        start_date = end_date - timedelta(days=command.days_back)
        prices = self._price_repo.get_history(
            symbol=command.symbol,
            start=start_date,
            end=end_date,
        )

        if not prices:
            logger.warning("No price data found for %s", command.symbol)
            empty_metrics = EvaluationMetricsResult(
                precision=0.0, recall=0.0, f1_score=0.0,
                true_positives=0, false_positives=0,
                false_negatives=0, support=0,
            )
            return EvaluateAnomaliesResult(
                symbol=command.symbol,
                total_detected=0,
                total_known=0,
                overall=empty_metrics,
                per_type=[],
            )

        # Generate ground-truth from statistical rules
        known = AnomalyEvaluator.generate_known_anomalies_from_data(prices)

        logger.info(
            "Generated %d ground-truth labels from %d price records",
            len(known),
            len(prices),
        )

        # Run backtest
        self._evaluator._tolerance = timedelta(days=command.date_tolerance_days)
        report = self._evaluator.backtest(
            symbol=command.symbol,
            prices=prices,
            known_anomalies=known,
        )

        # Map to output DTO
        overall = EvaluationMetricsResult(
            precision=round(report.overall.precision, 4),
            recall=round(report.overall.recall, 4),
            f1_score=round(report.overall.f1_score, 4),
            true_positives=report.overall.true_positives,
            false_positives=report.overall.false_positives,
            false_negatives=report.overall.false_negatives,
            support=report.overall.support,
        )

        per_type = [
            PerTypeMetricsResult(
                anomaly_type=atype,
                precision=round(m.precision, 4),
                recall=round(m.recall, 4),
                f1_score=round(m.f1_score, 4),
                support=m.support,
            )
            for atype, m in sorted(report.per_type.items())
        ]

        return EvaluateAnomaliesResult(
            symbol=command.symbol,
            total_detected=report.total_detected,
            total_known=report.total_known,
            overall=overall,
            per_type=per_type,
        )
