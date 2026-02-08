"""
Tests for the Sentiment Analysis module — all 5 gaps.

Gap 1: 3+ Tunisian news sources (spiders)
Gap 2: Article ↔ Symbol linkage (keyword matcher + repository)
Gap 3: get_sentiment SQL filters by symbol (adapter fix)
Gap 4: Daily aggregation pipeline (AggregateDailySentimentUseCase)
Gap 5: Tests (this file)

All tests use mocks / in-memory objects — no real DB or NLP model needed.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from app.domain.trading.entities import (
    ArticleSentiment,
    ScrapedArticle,
    Sentiment,
    SentimentScore,
)
from app.domain.trading.article_symbol_matcher import (
    BVMT_SYMBOL_ALIASES,
    ArticleSymbolMatcher,
    SymbolMatch,
)


# ===================================================================
# Gap 1 — Three real Tunisian news sources
# ===================================================================


class TestSpiderCoverage:
    """Verify that we have 3+ distinct Tunisian news sources."""

    def test_ilboursa_spider_exists(self):
        """IlBoursa spider is importable and targets ilboursa.com."""
        from scraping.spiders.ilboursa_spider import IlboursaSpider

        assert IlboursaSpider.name == "ilboursa"
        assert "ilboursa.com" in IlboursaSpider.allowed_domains

    def test_millim_spider_exists(self):
        """Millim spider is importable and targets millim.tn."""
        from scraping.spiders.millim_spider import MillimSpider

        assert MillimSpider.name == "millim"
        assert "millim.tn" in MillimSpider.allowed_domains

    def test_tustex_spider_exists(self):
        """Tustex spider is importable and targets tustex.com."""
        from scraping.spiders.tustex_spider import TustexSpider

        assert TustexSpider.name == "tustex"
        assert "tustex.com" in TustexSpider.allowed_domains

    def test_at_least_three_real_sources(self):
        """Confirm at least 3 distinct real news sources (excluding example)."""
        from scraping.spiders.ilboursa_spider import IlboursaSpider
        from scraping.spiders.millim_spider import MillimSpider
        from scraping.spiders.tustex_spider import TustexSpider

        domains = set()
        for spider_cls in [IlboursaSpider, MillimSpider, TustexSpider]:
            domains.update(spider_cls.allowed_domains)

        # Remove example.com if present
        domains.discard("example.com")
        assert len(domains) >= 3, f"Only {len(domains)} real domains: {domains}"

    def test_tustex_has_multiple_start_urls(self):
        """Tustex spider crawls multiple sections for broader coverage."""
        from scraping.spiders.tustex_spider import TustexSpider

        assert len(TustexSpider.start_urls) >= 2

    def test_tustex_has_parse_article_method(self):
        """Tustex spider has a parse_article callback."""
        from scraping.spiders.tustex_spider import TustexSpider

        assert hasattr(TustexSpider, "parse_article")
        assert callable(TustexSpider.parse_article)

    def test_ilboursa_historical_spider_exists(self):
        """Historical spider for deep date-range scraping."""
        from scraping.spiders.ilboursa_historical import IlboursaHistoricalSpider

        assert IlboursaHistoricalSpider.name == "ilboursa_historical"


# ===================================================================
# Gap 2 — Article ↔ Symbol linkage (domain matcher)
# ===================================================================


class TestArticleSymbolMatcher:
    """Test the pure-domain ArticleSymbolMatcher."""

    def test_matches_biat_by_full_name(self):
        """Match BIAT when the article mentions its full name."""
        matcher = ArticleSymbolMatcher()
        text = (
            "La Banque Internationale Arabe de Tunisie a annoncé "
            "des résultats exceptionnels pour le trimestre."
        )
        matches = matcher.match(text)
        symbols = [m.symbol for m in matches]
        assert "BIAT" in symbols

    def test_matches_biat_by_ticker(self):
        """Match BIAT by the short ticker alias."""
        matcher = ArticleSymbolMatcher()
        matches = matcher.match("Les actions de BIAT ont augmenté de 3%.")
        assert any(m.symbol == "BIAT" for m in matches)

    def test_matches_sfbt(self):
        """Match SFBT by its full company name."""
        matcher = ArticleSymbolMatcher()
        text = "La Société Frigorifique et Brasserie de Tunis publie ses comptes."
        matches = matcher.match(text)
        assert any(m.symbol == "SFBT" for m in matches)

    def test_matches_attijari(self):
        """Match ATTIJARI BANK by alias."""
        matcher = ArticleSymbolMatcher()
        text = "Attijari Bank enregistre une hausse de 15%."
        matches = matcher.match(text)
        assert any(m.symbol == "ATTIJARI BANK" for m in matches)

    def test_matches_multiple_symbols(self):
        """An article mentioning two companies returns both."""
        matcher = ArticleSymbolMatcher()
        text = (
            "BIAT et SFBT ont dominé la séance boursière. "
            "La BIAT progresse de 2% tandis que SFBT recule."
        )
        matches = matcher.match(text)
        symbols = {m.symbol for m in matches}
        assert "BIAT" in symbols
        assert "SFBT" in symbols

    def test_match_count_ordering(self):
        """Results are sorted by match_count descending."""
        matcher = ArticleSymbolMatcher()
        text = "BIAT BIAT BIAT. SFBT recule."  # BIAT mentioned 3x, SFBT 1x
        matches = matcher.match(text)
        assert len(matches) >= 2
        assert matches[0].symbol == "BIAT"
        assert matches[0].match_count >= matches[1].match_count

    def test_no_match_on_unrelated_text(self):
        """Generic news with no BVMT mentions returns empty."""
        matcher = ArticleSymbolMatcher()
        text = "Le temps sera ensoleillé sur la plupart des régions."
        matches = matcher.match(text)
        assert matches == []

    def test_empty_text_returns_empty(self):
        """Empty or blank text returns empty list."""
        matcher = ArticleSymbolMatcher()
        assert matcher.match("") == []
        assert matcher.match("   ") == []

    def test_match_single_returns_primary(self):
        """match_single returns the most-mentioned symbol."""
        matcher = ArticleSymbolMatcher()
        text = "BIAT enregistre de bons résultats. BIAT progresse."
        assert matcher.match_single(text) == "BIAT"

    def test_match_single_none_when_no_match(self):
        """match_single returns None when no symbol found."""
        matcher = ArticleSymbolMatcher()
        assert matcher.match_single("Pas de mention boursière.") is None

    def test_case_insensitive_matching(self):
        """Matching is case-insensitive."""
        matcher = ArticleSymbolMatcher()
        matches = matcher.match("biat a publié ses résultats.")
        assert any(m.symbol == "BIAT" for m in matches)

    def test_extra_aliases(self):
        """Extra aliases provided at init are also matched."""
        matcher = ArticleSymbolMatcher(
            extra_aliases={"NEWCO": ["nouvelle société", "newco"]}
        )
        text = "La Nouvelle Société a été créée en 2025."
        matches = matcher.match(text)
        assert any(m.symbol == "NEWCO" for m in matches)

    def test_symbol_aliases_dict_has_top_stocks(self):
        """The alias dictionary includes at least the top 5 BVMT stocks."""
        top_5 = {"BIAT", "SFBT", "BT", "ATTIJARI BANK", "SAH"}
        assert top_5.issubset(set(BVMT_SYMBOL_ALIASES.keys()))

    def test_poulina_match(self):
        """Match Poulina Group Holding."""
        matcher = ArticleSymbolMatcher()
        text = "Poulina Group Holding annonce un dividende exceptionnel."
        matches = matcher.match(text)
        assert any(m.symbol == "POULINA" for m in matches)

    def test_delice_match_with_accent(self):
        """Match Délice Holding (with French accent)."""
        matcher = ArticleSymbolMatcher()
        text = "Délice Holding maintient sa croissance."
        matches = matcher.match(text)
        assert any(m.symbol == "DELICE" for m in matches)

    def test_tunisair_match(self):
        """Match Tunisair."""
        matcher = ArticleSymbolMatcher()
        text = "Tunisair prévoit de nouvelles lignes aériennes."
        matches = matcher.match(text)
        assert any(m.symbol == "TUNISAIR" for m in matches)


# ===================================================================
# Gap 2 — Article ↔ Symbol linking use case
# ===================================================================


class TestLinkArticleSymbolsUseCase:
    """Test the LinkArticleSymbolsUseCase with mocked repos."""

    def _make_article(self, aid: int, title: str, content: str) -> ScrapedArticle:
        return ScrapedArticle(
            id=aid,
            url=f"https://example.com/article/{aid}",
            title=title,
            summary=None,
            content=content,
            published_at=datetime(2024, 6, 15, 10, 0),
        )

    def test_links_articles_to_symbols(self):
        """Use case matches articles to symbols and saves links."""
        from app.application.trading.link_article_symbols import (
            LinkArticleSymbolsCommand,
            LinkArticleSymbolsUseCase,
        )

        articles = [
            self._make_article(1, "BIAT résultats", "BIAT annonce de bons résultats."),
            self._make_article(2, "SFBT dividende", "SFBT verse un dividende."),
        ]

        article_repo = MagicMock()
        article_repo.get_unanalyzed_articles.return_value = articles
        article_repo.get_article_text.side_effect = lambda a: a.content or ""

        symbol_repo = MagicMock()
        symbol_repo.get_unlinked_article_ids.return_value = [1, 2]
        symbol_repo.save_batch.return_value = 2

        use_case = LinkArticleSymbolsUseCase(
            article_repo=article_repo,
            symbol_repo=symbol_repo,
        )

        result = use_case.execute(LinkArticleSymbolsCommand(batch_size=100))

        assert result.articles_scanned == 2
        assert result.links_created >= 2
        symbol_repo.save_batch.assert_called_once()

    def test_no_unlinked_articles(self):
        """When all articles are already linked, result is zero."""
        from app.application.trading.link_article_symbols import (
            LinkArticleSymbolsCommand,
            LinkArticleSymbolsUseCase,
        )

        article_repo = MagicMock()
        symbol_repo = MagicMock()
        symbol_repo.get_unlinked_article_ids.return_value = []

        use_case = LinkArticleSymbolsUseCase(
            article_repo=article_repo,
            symbol_repo=symbol_repo,
        )

        result = use_case.execute(LinkArticleSymbolsCommand(batch_size=100))

        assert result.articles_scanned == 0
        assert result.links_created == 0

    def test_articles_with_no_symbol_match(self):
        """Articles that don't mention any BVMT symbol are counted as no_match."""
        from app.application.trading.link_article_symbols import (
            LinkArticleSymbolsCommand,
            LinkArticleSymbolsUseCase,
        )

        articles = [
            self._make_article(
                1, "Météo Tunisie", "Le temps sera ensoleillé demain."
            ),
        ]

        article_repo = MagicMock()
        article_repo.get_unanalyzed_articles.return_value = articles
        article_repo.get_article_text.side_effect = lambda a: a.content or ""

        symbol_repo = MagicMock()
        symbol_repo.get_unlinked_article_ids.return_value = [1]

        use_case = LinkArticleSymbolsUseCase(
            article_repo=article_repo,
            symbol_repo=symbol_repo,
        )

        result = use_case.execute(LinkArticleSymbolsCommand(batch_size=100))

        assert result.articles_with_no_match >= 1
        assert result.links_created == 0


# ===================================================================
# Gap 3 — get_sentiment filters by symbol
# ===================================================================


class TestSentimentAdapterSymbolFilter:
    """Verify the adapter's get_sentiment query now filters by symbol.

    These tests read the adapter *source file* directly so they work even
    when the heavy ``transformers`` dependency is not installed.
    """

    @staticmethod
    def _adapter_source() -> str:
        """Return the raw source code of the adapter module."""
        from pathlib import Path

        src = Path(
            "c:/Users/mhached/Desktop/hack1/fixtrade/"
            "app/infrastructure/trading/sentiment_analysis_adapter.py"
        )
        return src.read_text(encoding="utf-8")

    def test_get_sentiment_sql_includes_symbol_filter(self):
        """The on-the-fly fallback query joins article_symbols and filters by symbol."""
        source = self._adapter_source()

        assert "article_symbols" in source, (
            "get_sentiment must JOIN article_symbols to filter by symbol"
        )
        assert "asym.symbol = :symbol" in source, (
            "get_sentiment must filter WHERE asym.symbol = :symbol"
        )

    def test_get_sentiment_tries_preaggregate_first(self):
        """The adapter first checks the sentiment_scores table."""
        source = self._adapter_source()
        assert "sentiment_scores" in source, (
            "get_sentiment should check sentiment_scores table first"
        )

    def test_get_sentiment_returns_score_for_symbol(self):
        """With a mock engine, get_sentiment returns a SentimentScore with the correct symbol.

        We mock ``app.nlp.sentiment`` before importing the adapter so the
        ``transformers`` heavy dependency is not required.
        """
        import importlib
        import sys
        import types

        # Provide a lightweight stub for the NLP module
        nlp_stub = types.ModuleType("app.nlp.sentiment")
        nlp_stub.SentimentAnalyzer = MagicMock  # type: ignore[attr-defined]
        sys.modules.setdefault("app.nlp", types.ModuleType("app.nlp"))
        sys.modules["app.nlp.sentiment"] = nlp_stub

        # Force (re-)import of the adapter with the stub in place
        mod_name = "app.infrastructure.trading.sentiment_analysis_adapter"
        sys.modules.pop(mod_name, None)
        adapter_mod = importlib.import_module(mod_name)
        SentimentAnalysisAdapter = adapter_mod.SentimentAnalysisAdapter

        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        # First query (pre-aggregated) returns None
        # Second query (on-the-fly) returns avg_score=0.5, count=3
        conn.execute.side_effect = [
            MagicMock(fetchone=MagicMock(return_value=None)),
            MagicMock(fetchone=MagicMock(return_value=(0.5, 3))),
        ]

        adapter = SentimentAnalysisAdapter(engine=engine)
        result = adapter.get_sentiment("BIAT", date(2024, 6, 15))

        assert result.symbol == "BIAT"
        assert result.article_count == 3
        assert result.sentiment == Sentiment.POSITIVE  # 0.5 > 0.3


# ===================================================================
# Gap 4 — Daily aggregation pipeline
# ===================================================================


class TestAggregateDailySentiment:
    """Test the AggregateDailySentimentUseCase with mocks."""

    def test_aggregation_upserts_scores(self):
        """Use case reads joined data and upserts to sentiment_scores."""
        from app.application.trading.aggregate_daily_sentiment import (
            AggregateDailySentimentCommand,
            AggregateDailySentimentUseCase,
        )

        # Mock the DB engine to return aggregated rows
        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        today = date.today()
        yesterday = today - timedelta(days=1)

        # Simulate aggregation query returning 2 rows
        conn.execute.return_value.fetchall.return_value = [
            ("BIAT", today, 0.6, 5),
            ("SFBT", yesterday, -0.4, 2),
        ]

        score_repo = MagicMock()

        use_case = AggregateDailySentimentUseCase(
            score_repo=score_repo,
            db_engine=engine,
        )

        result = use_case.execute(
            AggregateDailySentimentCommand(days_back=7)
        )

        assert result.symbols_processed == 2
        assert result.scores_upserted == 2
        assert score_repo.save.call_count == 2

        # Check the symbols in the saved scores
        saved_symbols = {
            call.args[0].symbol for call in score_repo.save.call_args_list
        }
        assert saved_symbols == {"BIAT", "SFBT"}

    def test_aggregation_with_symbol_filter(self):
        """Use case filters by symbol when provided."""
        from app.application.trading.aggregate_daily_sentiment import (
            AggregateDailySentimentCommand,
            AggregateDailySentimentUseCase,
        )

        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        conn.execute.return_value.fetchall.return_value = [
            ("BIAT", date.today(), 0.8, 10),
        ]

        score_repo = MagicMock()

        use_case = AggregateDailySentimentUseCase(
            score_repo=score_repo,
            db_engine=engine,
        )

        result = use_case.execute(
            AggregateDailySentimentCommand(symbol="BIAT", days_back=30)
        )

        assert result.symbols_processed == 1
        assert result.scores_upserted == 1
        # Verify the query was called with symbol parameter
        call_args = conn.execute.call_args
        assert "BIAT" in str(call_args)

    def test_aggregation_empty_returns_zero(self):
        """When no articles match, returns zero counts."""
        from app.application.trading.aggregate_daily_sentiment import (
            AggregateDailySentimentCommand,
            AggregateDailySentimentUseCase,
        )

        engine = MagicMock()
        conn = MagicMock()
        engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
        engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        conn.execute.return_value.fetchall.return_value = []

        score_repo = MagicMock()

        use_case = AggregateDailySentimentUseCase(
            score_repo=score_repo,
            db_engine=engine,
        )

        result = use_case.execute(
            AggregateDailySentimentCommand(days_back=7)
        )

        assert result.symbols_processed == 0
        assert result.scores_upserted == 0
        score_repo.save.assert_not_called()

    def test_sentiment_classification_thresholds(self):
        """Verify _avg_to_sentiment thresholds: >0.3=POS, <-0.3=NEG, else=NEU."""
        from app.application.trading.aggregate_daily_sentiment import (
            _avg_to_sentiment,
        )

        assert _avg_to_sentiment(0.5) == Sentiment.POSITIVE
        assert _avg_to_sentiment(0.31) == Sentiment.POSITIVE
        assert _avg_to_sentiment(0.3) == Sentiment.NEUTRAL
        assert _avg_to_sentiment(0.0) == Sentiment.NEUTRAL
        assert _avg_to_sentiment(-0.3) == Sentiment.NEUTRAL
        assert _avg_to_sentiment(-0.31) == Sentiment.NEGATIVE
        assert _avg_to_sentiment(-0.8) == Sentiment.NEGATIVE


# ===================================================================
# Ports — new ports exist
# ===================================================================


class TestNewPortsExist:
    """Verify that the new port ABCs are defined."""

    def test_article_symbol_repository_port(self):
        """ArticleSymbolRepository ABC is importable with expected methods."""
        from app.domain.trading.ports import ArticleSymbolRepository

        assert hasattr(ArticleSymbolRepository, "save")
        assert hasattr(ArticleSymbolRepository, "save_batch")
        assert hasattr(ArticleSymbolRepository, "get_symbols_for_article")
        assert hasattr(ArticleSymbolRepository, "get_articles_for_symbol")
        assert hasattr(ArticleSymbolRepository, "get_unlinked_article_ids")

    def test_sentiment_score_repository_port(self):
        """SentimentScoreRepository ABC is importable with expected methods."""
        from app.domain.trading.ports import SentimentScoreRepository

        assert hasattr(SentimentScoreRepository, "save")
        assert hasattr(SentimentScoreRepository, "get")
        assert hasattr(SentimentScoreRepository, "get_range")


# ===================================================================
# Infrastructure adapters — importable
# ===================================================================


class TestNewAdaptersExist:
    """Verify that the new infrastructure adapters are importable."""

    def test_article_symbol_repository_adapter(self):
        from app.infrastructure.trading.article_symbol_repository import (
            ArticleSymbolRepositoryAdapter,
        )

        assert ArticleSymbolRepositoryAdapter is not None

    def test_sentiment_score_repository_adapter(self):
        from app.infrastructure.trading.sentiment_score_repository import (
            SentimentScoreRepositoryAdapter,
        )

        assert SentimentScoreRepositoryAdapter is not None


# ===================================================================
# Schemas — new request/response schemas
# ===================================================================


class TestNewSchemas:
    """Verify new Pydantic schemas for the API layer."""

    def test_link_article_symbols_request(self):
        from app.interfaces.trading.schemas import LinkArticleSymbolsRequest

        req = LinkArticleSymbolsRequest(batch_size=100)
        assert req.batch_size == 100

    def test_link_article_symbols_response(self):
        from app.interfaces.trading.schemas import LinkArticleSymbolsResponse

        resp = LinkArticleSymbolsResponse(
            articles_scanned=10, links_created=5, articles_with_no_match=3
        )
        assert resp.links_created == 5

    def test_aggregate_daily_sentiment_request(self):
        from app.interfaces.trading.schemas import AggregateDailySentimentRequest

        req = AggregateDailySentimentRequest(symbol="BIAT", days_back=7)
        assert req.symbol == "BIAT"
        assert req.days_back == 7

    def test_aggregate_daily_sentiment_response(self):
        from app.interfaces.trading.schemas import (
            AggregateDailySentimentResponse,
            DailyScoreItemSchema,
        )

        resp = AggregateDailySentimentResponse(
            symbols_processed=1,
            dates_processed=1,
            scores_upserted=1,
            scores=[
                DailyScoreItemSchema(
                    symbol="BIAT",
                    score_date=date.today(),
                    score=Decimal("0.6"),
                    sentiment="positive",
                    article_count=5,
                )
            ],
        )
        assert resp.scores_upserted == 1
        assert resp.scores[0].symbol == "BIAT"


# ===================================================================
# Router endpoints — new endpoints registered
# ===================================================================


class TestNewRouterEndpoints:
    """Verify new API endpoints are registered in the router source."""

    @staticmethod
    def _router_source() -> str:
        from pathlib import Path

        src = Path(
            "c:/Users/mhached/Desktop/hack1/fixtrade/"
            "app/interfaces/trading/router.py"
        )
        return src.read_text(encoding="utf-8")

    def test_link_symbols_endpoint_exists(self):
        source = self._router_source()
        assert "/sentiment/link-symbols" in source, (
            "Router must declare a /sentiment/link-symbols endpoint"
        )
        assert "link_article_symbols" in source, (
            "Router must have a link_article_symbols handler function"
        )

    def test_aggregate_endpoint_exists(self):
        source = self._router_source()
        assert "/sentiment/aggregate" in source, (
            "Router must declare a /sentiment/aggregate endpoint"
        )
        assert "aggregate_daily_sentiment" in source, (
            "Router must have an aggregate_daily_sentiment handler function"
        )


# ===================================================================
# DB migration file exists
# ===================================================================


class TestMigrationFile:
    """Verify the migration SQL file for article_symbols exists."""

    def test_migration_003_exists(self):
        from pathlib import Path

        migration = Path(
            "c:/Users/mhached/Desktop/hack1/fixtrade/db/003_article_symbols.sql"
        )
        assert migration.exists(), "003_article_symbols.sql must exist"

    def test_migration_003_creates_article_symbols_table(self):
        from pathlib import Path

        migration = Path(
            "c:/Users/mhached/Desktop/hack1/fixtrade/db/003_article_symbols.sql"
        )
        content = migration.read_text()
        assert "article_symbols" in content
        assert "article_id" in content
        assert "symbol" in content
        assert "match_method" in content
