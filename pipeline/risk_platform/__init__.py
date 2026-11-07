"""Point-in-time risk data platform extension: PostgreSQL-backed fill
ledger, as-of position/exposure reconstruction, and end-of-day
reconciliation against an independent replay.

Everything in this package is synthetic, seeded and deterministic, same
discipline as the SQLite-backed options and commodity books above (see the
top-level README's "Honest framing" section). This package is the first
part of this repository that talks to a real client/server RDBMS, because
the as-of queries below are meant to be run as aggregate SQL against an
append-only ledger at a scale (2M+ rows) SQLite was never asked to carry
here; see the README for why SQLite was the right choice for the rest of
this repo and Postgres is the right choice for this one piece.
"""
