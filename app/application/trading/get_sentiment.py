"""
Use case: Retrieve sentiment analysis for a BVMT-listed symbol.

Input: GetSentimentQuery (symbol, optional target_date)
Output: SentimentResult
Side effects: None.
Failure cases: SymbolNotFoundError.
"""

import logging

from app.application.trading.dtos import GetSentimentQuery, SentimentResult
from app.domain.trading.ports import SentimentAnalysisPort

logger = logging.getLogger(__name__)


class GetSentimentUseCase:
    """Orchestrates sentiment retrieval for a given stock symbol.

    Delegates to the SentimentAnalysisPort for NLP inference
    and maps the domain entity to an application DTO.
    """

    def __init__(self, sentiment_port: SentimentAnalysisPort) -> None:
        self._sentiment_port = sentiment_port

    def execute(self, query: GetSentimentQuery) -> SentimentResult:
        """Run the sentiment retrieval use case.

        Args:
            query: The sentiment request containing symbol and optional date.

        Returns:
            Aggregated sentiment result for the symbol.
        """
        logger.info(
            "Retrieving sentiment for symbol=%s, date=%s",
            query.symbol,
            query.target_date,
        )

        # TODO: call the sentiment port and map result to DTO
        score = self._sentiment_port.get_sentiment(
            symbol=query.symbol,
            target_date=query.target_date,
        )

        return SentimentResult(
            symbol=score.symbol,
            date=score.date,
            score=score.score,
            sentiment=score.sentiment.value,
            article_count=score.article_count,
        )
