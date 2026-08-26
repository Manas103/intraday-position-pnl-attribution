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

-- Extension: commodity forward book (see README "Extension" section and
-- pipeline/commodity_*.py). Same idempotency discipline as `fills` above,
-- keyed on (deal_id, sequence). deal_id is NOT unique on its own: an amend
-- or cancel of an existing deal reuses the original deal_id with a new,
-- later sequence number.
CREATE TABLE IF NOT EXISTS commodity_deals (
    deal_id              TEXT    NOT NULL,
    sequence             INTEGER NOT NULL,
    action               TEXT    NOT NULL,   -- "new" | "amend" | "cancel"
    commodity            TEXT    NOT NULL,
    delivery_tenor_days  INTEGER NOT NULL,   -- identifies the instrument together with commodity
    volume               REAL    NOT NULL,   -- signed; "amend" carries the new ABSOLUTE volume, not a delta
    trade_price          REAL,               -- negotiated price, "new" deals only
    direction             TEXT    NOT NULL,  -- "buy" | "sell", derived from volume's sign
    session_id           INTEGER NOT NULL,
    timestamp            TEXT    NOT NULL,
    received_at          INTEGER NOT NULL,
    PRIMARY KEY (deal_id, sequence)
);

-- One row per instrument per session: the position rebuilt by applying all
-- accepted deals in sequence order (last write wins per deal_id), filled
-- forward on sessions with no activity for that instrument.
CREATE TABLE IF NOT EXISTS commodity_positions (
    instrument_id  TEXT    NOT NULL,
    session_id     INTEGER NOT NULL,
    quantity       REAL    NOT NULL,
    PRIMARY KEY (instrument_id, session_id)
);

-- One row per instrument per session: the P&L attribution decomposition.
-- price_pnl + curve_shift_pnl + time_pnl + new_deals_pnl + volume_pnl +
-- residual == actual_pnl by construction. Computed in-memory by
-- pipeline/commodity_attribution.py; this table documents the shape of
-- that result and is not currently populated by any script (matching the
-- pre-existing pnl_attribution table's own status in this schema).
CREATE TABLE IF NOT EXISTS commodity_pnl_attribution (
    instrument_id     TEXT    NOT NULL,
    session_id        INTEGER NOT NULL,
    mark_sod          REAL    NOT NULL,
    mark_eod          REAL    NOT NULL,
    actual_pnl        REAL    NOT NULL,
    price_pnl         REAL    NOT NULL,
    curve_shift_pnl   REAL    NOT NULL,
    time_pnl          REAL    NOT NULL,
    new_deals_pnl     REAL    NOT NULL,
    volume_pnl        REAL    NOT NULL,
    residual          REAL    NOT NULL,
    PRIMARY KEY (instrument_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_commodity_deals_instrument ON commodity_deals (commodity, delivery_tenor_days, sequence);
