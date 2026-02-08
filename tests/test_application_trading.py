"""
Tests for the trading application layer (use cases).

Tests use cases with mocked ports. No real infrastructure needed.
Each test verifies orchestration logic, not business rules.
"""


class TestPredictPriceUseCase:
    """Tests for the PredictPriceUseCase."""

    def test_valid_prediction_request(self) -> None:
        """Use case returns predictions for a valid request."""
        # TODO: mock PricePredictionPort and verify orchestration
        pass

    def test_invalid_horizon_raises_error(self) -> None:
        """Use case raises InvalidHorizonError for horizon outside 1-5."""
        # TODO: verify InvalidHorizonError is raised
        pass


class TestGetSentimentUseCase:
    """Tests for the GetSentimentUseCase."""

    def test_valid_sentiment_request(self) -> None:
        """Use case returns sentiment for a valid request."""
        # TODO: mock SentimentAnalysisPort and verify orchestration
        pass


class TestDetectAnomaliesUseCase:
    """Tests for the DetectAnomaliesUseCase."""

    def test_valid_anomaly_detection(self) -> None:
        """Use case returns anomalies for a valid request."""
        # TODO: mock AnomalyDetectionPort and verify orchestration
        pass


class TestGetRecommendationUseCase:
    """Tests for the GetRecommendationUseCase."""

    def test_valid_recommendation_request(self) -> None:
        """Use case returns recommendation for a valid request."""
        # TODO: mock ports and verify orchestration
        pass

    def test_missing_portfolio_raises_error(self) -> None:
        """Use case raises PortfolioNotFoundError for missing portfolio."""
        # TODO: mock PortfolioRepository returning None
        pass
