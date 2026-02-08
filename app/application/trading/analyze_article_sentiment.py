"""
Use case: Analyze sentiment of unanalyzed scraped articles.

Input: AnalyzeArticleSentimentCommand (batch_size)
Output: AnalyzeArticleSentimentResult (summary + per-article results)
Side effects: Persists sentiment results to article_sentiments table.
Failure cases:
    - No unanalyzed articles found (returns empty result).
    - Individual article analysis failures are logged and counted, not raised.
"""

import logging
from decimal import Decimal

from app.application.trading.dtos import (
    AnalyzeArticleSentimentCommand,
    AnalyzeArticleSentimentResult,
    ArticleSentimentResult,
)
from app.domain.trading.entities import ArticleSentiment
from app.domain.trading.ports import (
    ArticleSentimentRepository,
    ScrapedArticleRepository,
    SentimentAnalysisPort,
)

logger = logging.getLogger(__name__)


class AnalyzeArticleSentimentUseCase:
    """Orchestrates sentiment analysis on scraped articles.

    Reads unanalyzed articles from the article repository,
    runs NLP inference via the sentiment port, and persists
    results via the article sentiment repository.
    """

    def __init__(
        self,
        article_repo: ScrapedArticleRepository,
        sentiment_repo: ArticleSentimentRepository,
        sentiment_port: SentimentAnalysisPort,
    ) -> None:
        self._article_repo = article_repo
        self._sentiment_repo = sentiment_repo
        self._sentiment_port = sentiment_port

    def execute(
        self, command: AnalyzeArticleSentimentCommand
    ) -> AnalyzeArticleSentimentResult:
        """Run sentiment analysis on a batch of unanalyzed articles.

        Args:
            command: Contains batch_size for how many articles to process.

        Returns:
            Summary of analysis results including per-article scores.
        """
        logger.info(
            "Starting article sentiment analysis, batch_size=%d",
            command.batch_size,
        )

        articles = self._article_repo.get_unanalyzed_articles(
            limit=command.batch_size
        )

        if not articles:
            logger.info("No unanalyzed articles found.")
            return AnalyzeArticleSentimentResult(
                total_analyzed=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                failed_count=0,
                results=[],
            )

        logger.info("Found %d unanalyzed articles.", len(articles))

        results: list[ArticleSentimentResult] = []
        sentiments_to_save: list[ArticleSentiment] = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        failed_count = 0

        for article in articles:
            text = self._article_repo.get_article_text(article)
            if not text or not text.strip():
                logger.warning(
                    "Article %d has no usable text, skipping.", article.id
                )
                failed_count += 1
                continue

            try:
                score = self._sentiment_port.analyze_text(text)
                label = _score_to_label(score)

                sentiment = ArticleSentiment(
                    article_id=article.id,
                    sentiment_label=label,
                    sentiment_score=score,
                    confidence=None,
                )
                sentiments_to_save.append(sentiment)

                result = ArticleSentimentResult(
                    article_id=article.id,
                    sentiment_label=label,
                    sentiment_score=score,
                    confidence=None,
                )
                results.append(result)

                if score == 1:
                    positive_count += 1
                elif score == -1:
                    negative_count += 1
                else:
                    neutral_count += 1

            except Exception:
                logger.exception(
                    "Failed to analyze article %d.", article.id
                )
                failed_count += 1

        if sentiments_to_save:
            self._sentiment_repo.save_batch(sentiments_to_save)
            logger.info(
                "Persisted %d sentiment results.", len(sentiments_to_save)
            )

        total = len(results)
        logger.info(
            "Analysis complete: %d analyzed, %d positive, %d negative, "
            "%d neutral, %d failed.",
            total,
            positive_count,
            negative_count,
            neutral_count,
            failed_count,
        )

        return AnalyzeArticleSentimentResult(
            total_analyzed=total,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            failed_count=failed_count,
            results=results,
        )


def _score_to_label(score: int) -> str:
    """Convert integer sentiment score to label string."""
    if score == 1:
        return "positive"
    if score == -1:
        return "negative"
    return "neutral"
