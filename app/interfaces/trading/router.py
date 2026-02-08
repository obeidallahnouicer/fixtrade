"""
FastAPI router for the trading bounded context.

All routes delegate to use cases. No business logic here.
Input validation is handled by Pydantic schemas.
Error mapping is handled by centralized error handlers.
"""

from fastapi import APIRouter, Depends

from app.application.trading.dtos import (
    AnalyzeArticleSentimentCommand,
    DetectAnomaliesQuery,
    GetRecentAnomaliesQuery,
    GetRecommendationQuery,
    GetSentimentQuery,
    PredictLiquidityCommand,
    PredictPriceCommand,
    PredictVolumeCommand,
)
from app.application.trading.analyze_article_sentiment import (
    AnalyzeArticleSentimentUseCase,
)
from app.application.trading.detect_anomalies import DetectAnomaliesUseCase
from app.application.trading.get_recent_anomalies import GetRecentAnomaliesUseCase
from app.application.trading.get_recommendation import GetRecommendationUseCase
from app.application.trading.get_sentiment import GetSentimentUseCase
from app.application.trading.predict_liquidity import PredictLiquidityUseCase
from app.application.trading.predict_price import PredictPriceUseCase
from app.application.trading.predict_volume import PredictVolumeUseCase
from app.interfaces.trading.dependencies import (
    get_analyze_article_sentiment_use_case,
    get_detect_anomalies_use_case,
    get_predict_liquidity_use_case,
    get_predict_price_use_case,
    get_predict_volume_use_case,
    get_recent_anomalies_use_case,
    get_recommendation_use_case,
    get_sentiment_use_case,
)
from app.interfaces.trading.schemas import (
    AnalyzeArticleSentimentRequest,
    AnalyzeArticleSentimentResponse,
    AnomalyItem,
    ArticleSentimentItem,
    DetectAnomaliesRequest,
    DetectAnomaliesResponse,
    ErrorResponse,
    GetRecentAnomaliesRequest,
    GetRecommendationRequest,
    GetSentimentRequest,
    PredictLiquidityItem,
    PredictLiquidityRequest,
    PredictLiquidityResponse,
    PredictPriceItem,
    PredictPriceRequest,
    PredictPriceResponse,
    PredictVolumeItem,
    PredictVolumeRequest,
    PredictVolumeResponse,
    RecommendationResponse,
    SentimentResponse,
)

router = APIRouter(prefix="/trading", tags=["trading"])


@router.post(
    "/predictions",
    response_model=PredictPriceResponse,
    responses={422: {"model": ErrorResponse}},
    summary="Predict stock prices",
    description="Predict closing prices for a BVMT stock over 1-5 trading days.",
)
def predict_price(
    request: PredictPriceRequest,
    use_case: PredictPriceUseCase = Depends(get_predict_price_use_case),
) -> PredictPriceResponse:
    """Predict stock prices for a given symbol and horizon."""
    command = PredictPriceCommand(
        symbol=request.symbol,
        horizon_days=request.horizon_days,
    )
    results = use_case.execute(command)
    return PredictPriceResponse(
        predictions=[
            PredictPriceItem(
                symbol=r.symbol,
                target_date=r.target_date,
                predicted_close=r.predicted_close,
                confidence_lower=r.confidence_lower,
                confidence_upper=r.confidence_upper,
            )
            for r in results
        ]
    )


@router.post(
    "/sentiment",
    response_model=SentimentResponse,
    responses={422: {"model": ErrorResponse}},
    summary="Get sentiment analysis",
    description="Retrieve aggregated sentiment for a BVMT stock symbol.",
)
def get_sentiment(
    request: GetSentimentRequest,
    use_case: GetSentimentUseCase = Depends(get_sentiment_use_case),
) -> SentimentResponse:
    """Get sentiment analysis for a given symbol."""
    query = GetSentimentQuery(
        symbol=request.symbol,
        target_date=request.target_date,
    )
    result = use_case.execute(query)
    return SentimentResponse(
        symbol=result.symbol,
        date=result.date,
        score=result.score,
        sentiment=result.sentiment,
        article_count=result.article_count,
    )


@router.post(
    "/anomalies",
    response_model=DetectAnomaliesResponse,
    responses={422: {"model": ErrorResponse}},
    summary="Detect market anomalies",
    description="Detect anomalies in market data for a BVMT stock symbol.",
)
def detect_anomalies(
    request: DetectAnomaliesRequest,
    use_case: DetectAnomaliesUseCase = Depends(get_detect_anomalies_use_case),
) -> DetectAnomaliesResponse:
    """Detect anomalies for a given symbol."""
    query = DetectAnomaliesQuery(symbol=request.symbol)
    results = use_case.execute(query)
    return DetectAnomaliesResponse(
        anomalies=[
            AnomalyItem(
                id=r.id,
                symbol=r.symbol,
                detected_at=r.detected_at,
                anomaly_type=r.anomaly_type,
                severity=r.severity,
                description=r.description,
            )
            for r in results
        ]
    )


@router.get(
    "/anomalies/recent",
    response_model=DetectAnomaliesResponse,
    responses={422: {"model": ErrorResponse}},
    summary="Get recent anomaly alerts",
    description="Retrieve recent market anomalies for monitoring and alerts.",
)
def get_recent_anomalies(
    symbol: str | None = None,
    limit: int = 10,
    hours_back: int = 24,
    use_case: GetRecentAnomaliesUseCase = Depends(get_recent_anomalies_use_case),
) -> DetectAnomaliesResponse:
    """Get recent anomaly alerts (supports filtering by symbol and time)."""
    query = GetRecentAnomaliesQuery(
        symbol=symbol,
        limit=limit,
        hours_back=hours_back,
    )
    results = use_case.execute(query)
    return DetectAnomaliesResponse(
        anomalies=[
            AnomalyItem(
                id=r.id,
                symbol=r.symbol,
                detected_at=r.detected_at,
                anomaly_type=r.anomaly_type,
                severity=r.severity,
                description=r.description,
            )
            for r in results
        ]
    )


@router.post(
    "/recommendations",
    response_model=RecommendationResponse,
    responses={422: {"model": ErrorResponse}},
    summary="Get trade recommendation",
    description="Get a buy/sell/hold recommendation for a BVMT stock.",
)
def get_recommendation(
    request: GetRecommendationRequest,
    use_case: GetRecommendationUseCase = Depends(get_recommendation_use_case),
) -> RecommendationResponse:
    """Get a trade recommendation for a given symbol and portfolio."""
    query = GetRecommendationQuery(
        symbol=request.symbol,
        portfolio_id=request.portfolio_id,
    )
    result = use_case.execute(query)
    return RecommendationResponse(
        symbol=result.symbol,
        action=result.action,
        confidence=result.confidence,
        reasoning=result.reasoning,
    )


@router.post(
    "/predictions/volume",
    response_model=PredictVolumeResponse,
    responses={422: {"model": ErrorResponse}},
    summary="Predict transaction volume",
    description="Predict daily transaction volume for a BVMT stock over 1-5 trading days.",
)
def predict_volume(
    request: PredictVolumeRequest,
    use_case: PredictVolumeUseCase = Depends(get_predict_volume_use_case),
) -> PredictVolumeResponse:
    """Predict daily transaction volume for a given symbol and horizon."""
    command = PredictVolumeCommand(
        symbol=request.symbol,
        horizon_days=request.horizon_days,
    )
    results = use_case.execute(command)
    return PredictVolumeResponse(
        predictions=[
            PredictVolumeItem(
                symbol=r.symbol,
                target_date=r.target_date,
                predicted_volume=r.predicted_volume,
            )
            for r in results
        ]
    )


@router.post(
    "/predictions/liquidity",
    response_model=PredictLiquidityResponse,
    responses={422: {"model": ErrorResponse}},
    summary="Predict liquidity probabilities",
    description="Predict probability of high/medium/low liquidity for a BVMT stock.",
)
def predict_liquidity(
    request: PredictLiquidityRequest,
    use_case: PredictLiquidityUseCase = Depends(get_predict_liquidity_use_case),
) -> PredictLiquidityResponse:
    """Predict liquidity tier probabilities for a given symbol."""
    command = PredictLiquidityCommand(
        symbol=request.symbol,
        horizon_days=request.horizon_days,
    )
    results = use_case.execute(command)
    return PredictLiquidityResponse(
        forecasts=[
            PredictLiquidityItem(
                symbol=r.symbol,
                target_date=r.target_date,
                prob_low=r.prob_low,
                prob_medium=r.prob_medium,
                prob_high=r.prob_high,
                predicted_tier=r.predicted_tier,
            )
            for r in results
        ]
    )


@router.post(
    "/sentiment/analyze",
    response_model=AnalyzeArticleSentimentResponse,
    responses={422: {"model": ErrorResponse}},
    summary="Analyze article sentiment",
    description="Run NLP sentiment analysis on unanalyzed scraped articles.",
)
def analyze_article_sentiment(
    request: AnalyzeArticleSentimentRequest,
    use_case: AnalyzeArticleSentimentUseCase = Depends(
        get_analyze_article_sentiment_use_case
    ),
) -> AnalyzeArticleSentimentResponse:
    """Analyze sentiment of unanalyzed articles."""
    command = AnalyzeArticleSentimentCommand(batch_size=request.batch_size)
    result = use_case.execute(command)
    return AnalyzeArticleSentimentResponse(
        total_analyzed=result.total_analyzed,
        positive_count=result.positive_count,
        negative_count=result.negative_count,
        neutral_count=result.neutral_count,
        failed_count=result.failed_count,
        results=[
            ArticleSentimentItem(
                article_id=r.article_id,
                sentiment_label=r.sentiment_label,
                sentiment_score=r.sentiment_score,
                confidence=r.confidence,
            )
            for r in result.results
        ],
    )

