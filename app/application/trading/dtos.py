"""
Data Transfer Objects for the trading application layer.

DTOs carry data between the interface and application layers.
They are plain dataclasses with no behavior.
"""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID


@dataclass(frozen=True)
class PredictPriceCommand:
    """Input DTO for requesting a price prediction.

    Attributes:
        symbol: BVMT stock ticker symbol.
        horizon_days: Number of future trading days to predict (1-5).
    """

    symbol: str
    horizon_days: int


@dataclass(frozen=True)
class PredictPriceResult:
    """Output DTO for a single predicted price point.

    Attributes:
        symbol: BVMT stock ticker symbol.
        target_date: The future date of this prediction.
        predicted_close: Predicted closing price.
        confidence_lower: Lower bound of the confidence interval.
        confidence_upper: Upper bound of the confidence interval.
    """

    symbol: str
    target_date: date
    predicted_close: Decimal
    confidence_lower: Decimal
    confidence_upper: Decimal


@dataclass(frozen=True)
class GetSentimentQuery:
    """Input DTO for retrieving sentiment for a symbol.

    Attributes:
        symbol: BVMT stock ticker symbol.
        target_date: Optional specific date to query. Defaults to today.
    """

    symbol: str
    target_date: date | None = None


@dataclass(frozen=True)
class SentimentResult:
    """Output DTO for a sentiment analysis result.

    Attributes:
        symbol: BVMT stock ticker symbol.
        date: The date this sentiment refers to.
        score: Numeric sentiment score.
        sentiment: Classification label (positive/negative/neutral).
        article_count: Number of articles analyzed.
    """

    symbol: str
    date: date
    score: Decimal
    sentiment: str
    article_count: int


@dataclass(frozen=True)
class DetectAnomaliesQuery:
    """Input DTO for requesting anomaly detection.

    Attributes:
        symbol: BVMT stock ticker symbol.
    """

    symbol: str


@dataclass(frozen=True)
class AnomalyResult:
    """Output DTO for a detected anomaly.

    Attributes:
        id: Unique identifier of the anomaly.
        symbol: BVMT stock ticker symbol.
        detected_at: When the anomaly was detected.
        anomaly_type: Category of anomaly (volume_spike, price_swing, etc.).
        severity: Severity score.
        description: Human-readable description.
    """

    id: UUID
    symbol: str
    detected_at: datetime
    anomaly_type: str
    severity: Decimal
    description: str


@dataclass(frozen=True)
class GetRecentAnomaliesQuery:
    """Input DTO for requesting recent anomaly alerts.

    Attributes:
        symbol: Optional filter by stock symbol.
        limit: Maximum number of alerts to return.
        hours_back: Number of hours to look back.
    """

    symbol: str | None = None
    limit: int = 10
    hours_back: int = 24


@dataclass(frozen=True)
class GetRecommendationQuery:
    """Input DTO for requesting a trade recommendation.

    Attributes:
        symbol: BVMT stock ticker symbol.
        portfolio_id: UUID of the portfolio to consider.
    """

    symbol: str
    portfolio_id: UUID


@dataclass(frozen=True)
class RecommendationResult:
    """Output DTO for a trade recommendation.

    Attributes:
        symbol: BVMT stock ticker symbol.
        action: Recommended action (buy/sell/hold).
        confidence: Confidence score of the recommendation.
        reasoning: Explanation in natural language.
    """

    symbol: str
    action: str
    confidence: Decimal
    reasoning: str


# ------------------------------------------------------------------
# Volume Prediction DTOs
# ------------------------------------------------------------------


@dataclass(frozen=True)
class PredictVolumeCommand:
    """Input DTO for requesting a volume prediction.

    Attributes:
        symbol: BVMT stock ticker symbol.
        horizon_days: Number of future trading days to predict (1-5).
    """

    symbol: str
    horizon_days: int


@dataclass(frozen=True)
class PredictVolumeResult:
    """Output DTO for a single predicted volume point.

    Attributes:
        symbol: BVMT stock ticker symbol.
        target_date: The future date of this prediction.
        predicted_volume: Predicted daily transaction volume.
    """

    symbol: str
    target_date: date
    predicted_volume: int


# ------------------------------------------------------------------
# Liquidity Probability DTOs
# ------------------------------------------------------------------


@dataclass(frozen=True)
class PredictLiquidityCommand:
    """Input DTO for requesting a liquidity probability forecast.

    Attributes:
        symbol: BVMT stock ticker symbol.
        horizon_days: Number of future trading days to predict (1-5).
    """

    symbol: str
    horizon_days: int


@dataclass(frozen=True)
class PredictLiquidityResult:
    """Output DTO for a liquidity probability forecast.

    Attributes:
        symbol: BVMT stock ticker symbol.
        target_date: The future date of this forecast.
        prob_low: Probability of low liquidity (volume < 1 000).
        prob_medium: Probability of medium liquidity (1 000 ≤ vol < 10 000).
        prob_high: Probability of high liquidity (volume ≥ 10 000).
        predicted_tier: The most likely liquidity tier.
    """

    symbol: str
    target_date: date
    prob_low: Decimal
    prob_medium: Decimal
    prob_high: Decimal
    predicted_tier: str


# ------------------------------------------------------------------
# Article Sentiment Analysis DTOs
# ------------------------------------------------------------------


@dataclass(frozen=True)
class AnalyzeArticleSentimentCommand:
    """Input DTO for triggering sentiment analysis on unanalyzed articles.

    Attributes:
        batch_size: Maximum number of articles to process in one run.
    """

    batch_size: int = 50


@dataclass(frozen=True)
class ArticleSentimentResult:
    """Output DTO for a single article's sentiment analysis.

    Attributes:
        article_id: Database ID of the scraped article.
        sentiment_label: Classification label (positive/negative/neutral).
        sentiment_score: Numeric score (-1, 0, 1).
        confidence: Model confidence (0.0-1.0), if available.
    """

    article_id: int
    sentiment_label: str
    sentiment_score: int
    confidence: Decimal | None


@dataclass(frozen=True)
class AnalyzeArticleSentimentResult:
    """Output DTO summarizing a batch sentiment analysis run.

    Attributes:
        total_analyzed: Number of articles analyzed in this run.
        positive_count: Number of articles classified as positive.
        negative_count: Number of articles classified as negative.
        neutral_count: Number of articles classified as neutral.
        failed_count: Number of articles that failed analysis.
        results: Per-article sentiment results.
    """

    total_analyzed: int
    positive_count: int
    negative_count: int
    neutral_count: int
    failed_count: int
    results: list[ArticleSentimentResult]


# ------------------------------------------------------------------
# Anomaly Evaluation DTOs
# ------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluateAnomaliesCommand:
    """Input DTO for running anomaly detection evaluation.

    Attributes:
        symbol: BVMT stock ticker symbol to evaluate.
        days_back: Number of historical days to use for backtesting.
        date_tolerance_days: Date tolerance for matching (±N days).
    """

    symbol: str
    days_back: int = 90
    date_tolerance_days: int = 1


@dataclass(frozen=True)
class EvaluationMetricsResult:
    """Output DTO for anomaly evaluation metrics.

    Attributes:
        precision: Precision score (0.0–1.0).
        recall: Recall score (0.0–1.0).
        f1_score: F1-Score (0.0–1.0).
        true_positives: Count of correctly detected anomalies.
        false_positives: Count of false alarms.
        false_negatives: Count of missed anomalies.
        support: Total ground-truth positives.
    """

    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    support: int


@dataclass(frozen=True)
class PerTypeMetricsResult:
    """Per anomaly-type metrics in the evaluation output."""

    anomaly_type: str
    precision: float
    recall: float
    f1_score: float
    support: int


@dataclass(frozen=True)
class EvaluateAnomaliesResult:
    """Output DTO for the full evaluation report.

    Attributes:
        symbol: Stock ticker evaluated.
        total_detected: Number of anomalies detected.
        total_known: Number of known/labeled anomalies.
        overall: Overall Precision/Recall/F1 metrics.
        per_type: Per anomaly-type breakdown.
    """

    symbol: str
    total_detected: int
    total_known: int
    overall: EvaluationMetricsResult
    per_type: list[PerTypeMetricsResult]


# ------------------------------------------------------------------
# Intraday Anomaly Detection DTOs
# ------------------------------------------------------------------


@dataclass(frozen=True)
class DetectIntradayAnomaliesCommand:
    """Input DTO for intraday anomaly detection.

    Attributes:
        symbol: BVMT stock ticker symbol.
        days_back: Number of recent trading days to scan (1-30).
    """

    symbol: str
    days_back: int = 5
