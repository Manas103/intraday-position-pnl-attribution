"""Idempotent, batched ingest of the fill stream into PostgreSQL.

COPY has no ON CONFLICT clause, so bulk-loading millions of rows directly
into a constrained table with one COPY per row (or even one big COPY) does
not give idempotency for free. The pattern used here: COPY each batch into
an unlogged staging table with no constraints (fast, no WAL), then one
`INSERT ... SELECT ... ON CONFLICT (fill_id, sequence) DO NOTHING` moves
whatever is genuinely new into `risk_fills` and reports exactly how many
rows that was (`INSERT ... RETURNING` row count), and the staging table is
truncated for the next batch. This is the standard efficient way to get
bulk-COPY throughput and row-level idempotency at the same time; see
sql/risk_platform_schema.sql for the staging table's own comment.
"""

from __future__ import annotations

import io
import os
from typing import Iterable

import psycopg2

from .simulator import Fill

DEFAULT_DSN = dict(
    host=os.environ.get("RISK_PG_HOST", "127.0.0.1"),
    port=int(os.environ.get("RISK_PG_PORT", "5432")),
    dbname=os.environ.get("RISK_PG_DB", "risk_platform"),
    user=os.environ.get("RISK_PG_USER", "postgres"),
    password=os.environ.get("RISK_PG_PASSWORD", "postgres"),
)


def connect(dsn: dict | None = None):
    return psycopg2.connect(**(dsn or DEFAULT_DSN))


def ensure_schema(conn) -> None:
    schema_path = os.path.join(os.path.dirname(__file__), "..", "..", "sql", "risk_platform_schema.sql")
    with open(schema_path) as f:
        ddl = f.read()
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def reset_tables(conn) -> None:
    """Truncates both tables. Used between the canonical and perturbed
    runs of the replay check, and at the start of each ingest benchmark,
    so every run starts from the same empty ledger rather than
    accumulating across invocations."""
    with conn.cursor() as cur:
        cur.execute("TRUNCATE risk_fills, risk_fills_staging;")
    conn.commit()


def _copy_batch(conn, batch: list[Fill]) -> None:
    buf = io.StringIO()
    for f in batch:
        buf.write(f"{f.fill_id}\t{f.sequence}\t{f.instrument}\t{f.quantity}\t{f.price}\t{f.session_id}\n")
    buf.seek(0)
    with conn.cursor() as cur:
        cur.copy_expert(
            "COPY risk_fills_staging (fill_id, sequence, instrument, quantity, price, session_id) "
            "FROM STDIN WITH (FORMAT text)",
            buf,
        )


def ingest_stream(conn, fills: Iterable[Fill], batch_size: int = 200_000) -> int:
    """Ingests `fills` in batches of `batch_size`. Returns the total number
    of rows actually accepted (i.e. that were not already present under
    the same (fill_id, sequence) key) across the whole stream."""
    accepted_total = 0
    batch: list[Fill] = []
    for f in fills:
        batch.append(f)
        if len(batch) >= batch_size:
            accepted_total += _flush_batch(conn, batch)
            batch = []
    if batch:
        accepted_total += _flush_batch(conn, batch)
    return accepted_total


def _flush_batch(conn, batch: list[Fill]) -> int:
    _copy_batch(conn, batch)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO risk_fills (fill_id, sequence, instrument, quantity, price, session_id)
            SELECT fill_id, sequence, instrument, quantity, price, session_id
            FROM risk_fills_staging
            ON CONFLICT (fill_id, sequence) DO NOTHING
            """
        )
        accepted = cur.rowcount
        cur.execute("TRUNCATE risk_fills_staging;")
    conn.commit()
    return accepted


def total_rows(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM risk_fills;")
        return cur.fetchone()[0]
