"""
Dependency injection for the trading bounded context.

Provides FastAPI dependency functions that wire infrastructure
adapters into use cases via constructor injection.
These are the composition root for the trading context.
"""

from app.application.trading.detect_anomalies import DetectAnomaliesUseCase
from app.application.trading.get_recommendation import GetRecommendationUseCase
from app.application.trading.get_sentiment import GetSentimentUseCase
from app.application.trading.predict_liquidity import PredictLiquidityUseCase
from app.application.trading.predict_price import PredictPriceUseCase
from app.application.trading.predict_volume import PredictVolumeUseCase
from app.infrastructure.trading.anomaly_detection_adapter import (
    AnomalyDetectionAdapter,
)
from app.infrastructure.trading.decision_engine_adapter import DecisionEngineAdapter
from app.infrastructure.trading.portfolio_repository import (
    PortfolioRepositoryAdapter,
)
from app.infrastructure.trading.price_prediction_adapter import (
    PricePredictionAdapter,
)
from app.infrastructure.trading.sentiment_analysis_adapter import (
    SentimentAnalysisAdapter,
)


def get_predict_price_use_case() -> PredictPriceUseCase:
    """Build PredictPriceUseCase with its infrastructure dependencies."""
    return PredictPriceUseCase(
        prediction_port=PricePredictionAdapter(),
    )


def get_predict_volume_use_case() -> PredictVolumeUseCase:
    """Build PredictVolumeUseCase with its infrastructure dependencies."""
    return PredictVolumeUseCase(
        prediction_port=PricePredictionAdapter(),
    )


def get_predict_liquidity_use_case() -> PredictLiquidityUseCase:
    """Build PredictLiquidityUseCase with its infrastructure dependencies."""
    return PredictLiquidityUseCase(
        prediction_port=PricePredictionAdapter(),
    )


def get_sentiment_use_case() -> GetSentimentUseCase:
    """Build GetSentimentUseCase with its infrastructure dependencies."""
    return GetSentimentUseCase(
        sentiment_port=SentimentAnalysisAdapter(),
    )


def get_detect_anomalies_use_case() -> DetectAnomaliesUseCase:
    """Build DetectAnomaliesUseCase with its infrastructure dependencies."""
    return DetectAnomaliesUseCase(
        anomaly_port=AnomalyDetectionAdapter(),
    )


def get_recommendation_use_case() -> GetRecommendationUseCase:
    """Build GetRecommendationUseCase with its infrastructure dependencies."""
    return GetRecommendationUseCase(
        portfolio_repo=PortfolioRepositoryAdapter(),
        decision_port=DecisionEngineAdapter(),
    )
