"""
Dependency injection for the trading bounded context.

Provides FastAPI dependency functions that wire infrastructure
adapters into use cases via constructor injection.
These are the composition root for the trading context.
"""

from sqlalchemy import create_engine

from app.application.trading.analyze_article_sentiment import (
    AnalyzeArticleSentimentUseCase,
)
from app.application.trading.detect_anomalies import DetectAnomaliesUseCase
from app.application.trading.get_recommendation import GetRecommendationUseCase
from app.application.trading.get_sentiment import GetSentimentUseCase
from app.application.trading.predict_liquidity import PredictLiquidityUseCase
from app.application.trading.predict_price import PredictPriceUseCase
from app.application.trading.predict_volume import PredictVolumeUseCase
from app.core.config import settings
from app.infrastructure.trading.anomaly_detection_adapter import (
    AnomalyDetectionAdapter,
)
from app.infrastructure.trading.article_sentiment_repository import (
    ArticleSentimentRepositoryAdapter,
)
from app.infrastructure.trading.decision_engine_adapter import DecisionEngineAdapter
from app.infrastructure.trading.portfolio_repository import (
    PortfolioRepositoryAdapter,
)
from app.infrastructure.trading.price_prediction_adapter import (
    PricePredictionAdapter,
)
from app.infrastructure.trading.scraped_article_repository import (
    ScrapedArticleRepositoryAdapter,
)
from app.infrastructure.trading.sentiment_analysis_adapter import (
    SentimentAnalysisAdapter,
)


def _get_db_engine():
    """Build a SQLAlchemy engine from application settings."""
    dsn = settings.get_scraping_postgres_dsn()
    return create_engine(dsn, pool_pre_ping=True)


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
    engine = _get_db_engine()
    return GetSentimentUseCase(
        sentiment_port=SentimentAnalysisAdapter(engine=engine),
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


def get_analyze_article_sentiment_use_case() -> AnalyzeArticleSentimentUseCase:
    """Build AnalyzeArticleSentimentUseCase with its infrastructure dependencies."""
    engine = _get_db_engine()
    return AnalyzeArticleSentimentUseCase(
        article_repo=ScrapedArticleRepositoryAdapter(engine=engine),
        sentiment_repo=ArticleSentimentRepositoryAdapter(engine=engine),
        sentiment_port=SentimentAnalysisAdapter(engine=engine),
    )
