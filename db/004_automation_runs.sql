-- Automation observability and idempotency support.

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              BIGSERIAL PRIMARY KEY,
    job_name        VARCHAR(50) NOT NULL,
    status          VARCHAR(20) NOT NULL
                        CHECK (status IN ('running', 'completed', 'failed')),
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ,
    details         JSONB NOT NULL DEFAULT '{}'::jsonb,
    error           TEXT
);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_job_started
    ON pipeline_runs (job_name, started_at DESC);

SELECT 'Migration 004 applied successfully' AS status;
