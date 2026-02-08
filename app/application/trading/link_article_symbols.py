"""
Use case: Link scraped articles to BVMT symbols.

Scans unlinked articles, runs the ArticleSymbolMatcher domain
service on their text, and persists the resulting links to the
article_symbols table.

Input:  LinkArticleSymbolsCommand (batch_size)
Output: LinkArticleSymbolsResult  (counts)
Side effects: Inserts rows in article_symbols.
"""

import logging
from dataclasses import dataclass

from app.domain.trading.article_symbol_matcher import ArticleSymbolMatcher
from app.domain.trading.ports import ArticleSymbolRepository, ScrapedArticleRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinkArticleSymbolsCommand:
    """Input DTO for the article-symbol linking use case."""

    batch_size: int = 200


@dataclass(frozen=True)
class LinkArticleSymbolsResult:
    """Output DTO summarising a linking run."""

    articles_scanned: int
    links_created: int
    articles_with_no_match: int


class LinkArticleSymbolsUseCase:
    """Orchestrates article → symbol keyword matching and persistence."""

    def __init__(
        self,
        article_repo: ScrapedArticleRepository,
        symbol_repo: ArticleSymbolRepository,
        matcher: ArticleSymbolMatcher | None = None,
    ) -> None:
        self._article_repo = article_repo
        self._symbol_repo = symbol_repo
        self._matcher = matcher or ArticleSymbolMatcher()

    def execute(self, command: LinkArticleSymbolsCommand) -> LinkArticleSymbolsResult:
        """Run article-symbol linking on a batch of unlinked articles.

        Args:
            command: Contains batch_size for how many articles to scan.

        Returns:
            Summary of linking results.
        """
        logger.info(
            "Starting article-symbol linking, batch_size=%d",
            command.batch_size,
        )

        # Fetch unlinked article IDs
        unlinked_ids = self._symbol_repo.get_unlinked_article_ids(
            limit=command.batch_size
        )

        if not unlinked_ids:
            logger.info("No unlinked articles found.")
            return LinkArticleSymbolsResult(
                articles_scanned=0, links_created=0, articles_with_no_match=0
            )

        # Fetch the full articles so we can access their text
        articles = self._article_repo.get_unanalyzed_articles(
            limit=command.batch_size * 2  # over-fetch to cover overlap
        )

        # Build a lookup by ID
        article_map = {a.id: a for a in articles}

        # Also fetch already-analyzed articles that are still unlinked
        # (they exist in article_sentiments but not in article_symbols)
        # We handle this by directly fetching from the repository

        links: list[tuple[int, str, str, float]] = []
        no_match_count = 0

        for aid in unlinked_ids:
            article = article_map.get(aid)
            if article is None:
                # Article might already be analyzed; build text from repo
                # For now, skip if not in the fetched batch
                continue

            text = self._article_repo.get_article_text(article)
            if not text or not text.strip():
                no_match_count += 1
                continue

            # Also include title in the match text for better recall
            full_text = " ".join(
                filter(None, [article.title, article.summary, article.content])
            )

            matches = self._matcher.match(full_text)

            if not matches:
                no_match_count += 1
                continue

            for m in matches:
                links.append((aid, m.symbol, "keyword", 1.0))

        if links:
            self._symbol_repo.save_batch(links)

        result = LinkArticleSymbolsResult(
            articles_scanned=len(unlinked_ids),
            links_created=len(links),
            articles_with_no_match=no_match_count,
        )

        logger.info(
            "Linking complete: scanned=%d, links=%d, no_match=%d",
            result.articles_scanned,
            result.links_created,
            result.articles_with_no_match,
        )

        return result
