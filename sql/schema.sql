-- Schema for the intraday position / P&L attribution ledger.
-- Backing store is SQLite (stdlib sqlite3), so the whole pipeline runs
-- offline with zero external services. See README for why SQLite over a
-- message broker or a client/server RDBMS for this project's scope.

-- Fills: the append-only record of what happened on the (simulated) market.
-- Idempotency is enforced at the storage layer: the UNIQUE constraint on
-- (fill_id, sequence) means replaying the same message twice is rejected
-- by the database itself, not just by application-level bookkeeping.
CREATE TABLE IF NOT EXISTS fills (
    fill_id      TEXT    NOT NULL,
    sequence     INTEGER NOT NULL,
    instrument   TEXT    NOT NULL,
    quantity     REAL    NOT NULL,   -- signed: positive = buy, negative = sell
    price        REAL    NOT NULL,
    session_id   INTEGER NOT NULL,
    received_at  INTEGER NOT NULL,   -- monotonic ingest-order counter, NOT a wall clock;
                                      -- used only for diagnostics, never for ordering positions
    PRIMARY KEY (fill_id, sequence)
);

-- One row per instrument per session: the position rebuilt by applying all
-- accepted fills for that instrument, in sequence order, up to that session.
CREATE TABLE IF NOT EXISTS positions (
    instrument   TEXT    NOT NULL,
    session_id   INTEGER NOT NULL,
    quantity     REAL    NOT NULL,
    avg_price    REAL    NOT NULL,
    PRIMARY KEY (instrument, session_id)
);

-- One row per instrument per session: the P&L attribution decomposition
-- produced by the C++ revaluation engine. delta_pnl + gamma_pnl + vega_pnl
-- + theta_pnl + residual == actual_pnl by construction (see
-- tests/test_attribution.py::test_components_sum_to_actual_pnl).
CREATE TABLE IF NOT EXISTS pnl_attribution (
    instrument   TEXT    NOT NULL,
    session_id   INTEGER NOT NULL,
    price_sod    REAL    NOT NULL,
    price_eod    REAL    NOT NULL,
    actual_pnl   REAL    NOT NULL,
    delta_pnl    REAL    NOT NULL,
    gamma_pnl    REAL    NOT NULL,
    vega_pnl     REAL    NOT NULL,
    theta_pnl    REAL    NOT NULL,
    residual     REAL    NOT NULL,
    PRIMARY KEY (instrument, session_id)
);

CREATE INDEX IF NOT EXISTS idx_fills_instrument ON fills (instrument, sequence);
