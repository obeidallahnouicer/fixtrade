"""
Adapter: Sentiment analysis NLP pipeline.

Implements SentimentAnalysisPort.
Responsible for running NLP inference on scraped news articles.
"""

from datetime import date
from typing import Optional

from app.domain.trading.entities import SentimentScore
from app.domain.trading.ports import SentimentAnalysisPort


class SentimentAnalysisAdapter(SentimentAnalysisPort):
    """Concrete adapter for NLP-based sentiment analysis.

    Implements the SentimentAnalysisPort defined in the domain layer.
    In production, this will query the NLP pipeline or sentiment store.
    """

    def __init__(self) -> None:
        # TODO: inject NLP model or sentiment data source
        pass

    def get_sentiment(
        self, symbol: str, target_date: Optional[date] = None
    ) -> SentimentScore:
        """Return aggregated sentiment score for a symbol on a given date.

        Args:
            symbol: BVMT stock ticker.
            target_date: Optional date for the sentiment query.

        Returns:
            Aggregated SentimentScore entity.
        """
        # TODO: run NLP inference or query sentiment store
        raise NotImplementedError("SentimentAnalysisAdapter.get_sentiment")
