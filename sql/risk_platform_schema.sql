-- Schema for the point-in-time risk data platform extension. Backing
-- store is PostgreSQL (the original options/commodity books above use
-- SQLite; this extension is the first piece of this repo that needs a
-- real client/server RDBMS, because the as-of queries below are
-- aggregate SQL run directly against an append-only ledger at a scale
-- SQLite was never asked to carry).
--
-- Idempotency is enforced the same way as the SQLite ledger above: a
-- UNIQUE constraint (here, the primary key) on (fill_id, sequence), so
-- replaying the same message twice is rejected by the database itself.
CREATE TABLE IF NOT EXISTS risk_fills (
    fill_id      TEXT      NOT NULL,
    sequence     BIGINT    NOT NULL,
    instrument   TEXT      NOT NULL,
    quantity     NUMERIC   NOT NULL,   -- signed: positive = buy, negative = sell
    price        NUMERIC   NOT NULL,
    session_id   INTEGER   NOT NULL,
    PRIMARY KEY (fill_id, sequence)
);

-- Every as-of query in pipeline/risk_platform/asof.py filters on
-- (instrument, sequence); this is the index that makes that filter not a
-- sequential scan once the table holds millions of rows.
CREATE INDEX IF NOT EXISTS idx_risk_fills_instrument_sequence
    ON risk_fills (instrument, sequence);

-- Unlogged staging table for the COPY-then-upsert ingest pattern in
-- pipeline/risk_platform/pg_ingest.py: COPY has no ON CONFLICT clause, so
-- bulk load lands here first (no constraints, no WAL, fast), then a single
-- INSERT ... SELECT ... ON CONFLICT DO NOTHING moves accepted rows into
-- risk_fills and the staging table is truncated for the next batch.
CREATE UNLOGGED TABLE IF NOT EXISTS risk_fills_staging (
    fill_id      TEXT,
    sequence     BIGINT,
    instrument   TEXT,
    quantity     NUMERIC,
    price        NUMERIC,
    session_id   INTEGER
);
