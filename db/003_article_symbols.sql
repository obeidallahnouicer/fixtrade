-- ============================================================
-- Migration 003: Article ↔ Symbol linkage
-- Links scraped articles to the BVMT symbols they mention.
-- ============================================================

-- ── Junction table: which articles mention which stocks ──────
CREATE TABLE IF NOT EXISTS article_symbols (
    id              BIGSERIAL PRIMARY KEY,
    article_id      BIGINT         NOT NULL REFERENCES scraped_articles(id) ON DELETE CASCADE,
    symbol          VARCHAR(50)    NOT NULL,
    match_method    VARCHAR(20)    NOT NULL DEFAULT 'keyword'
                        CHECK (match_method IN ('keyword', 'ner', 'manual')),
    confidence      NUMERIC(5,4)   DEFAULT 1.0,
    created_at      TIMESTAMPTZ    DEFAULT NOW(),

    CONSTRAINT uq_article_symbol UNIQUE (article_id, symbol)
);

CREATE INDEX IF NOT EXISTS idx_article_symbols_article
    ON article_symbols (article_id);
CREATE INDEX IF NOT EXISTS idx_article_symbols_symbol
    ON article_symbols (symbol);
CREATE INDEX IF NOT EXISTS idx_article_symbols_symbol_article
    ON article_symbols (symbol, article_id);
