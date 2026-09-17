"""SQLite persistence for the watchlist feature.

Two tables:

- ``watchlist``: one row per (sneaker_id, size) pair a user is tracking.
  ``UNIQUE(sneaker_id, size)`` means adding the same pair twice is a
  conflict (surfaced by routes.py as 409), not a duplicate row.
- ``snapshots``: one row per priced point in time for a watchlist item,
  so the frontend can chart price-over-time. ``ON DELETE CASCADE`` means
  removing a watchlist item removes its snapshots automatically -- but
  only because every connection enables ``PRAGMA foreign_keys=ON``
  (SQLite does not persist that pragma across connections, so it must be
  set every time one is opened, not just once at init time).

Connection handling follows the standard Flask/SQLite pattern: one
connection per request, stashed on `flask.g` by `get_db()` and closed by
`close_db()` (registered as a `teardown_appcontext` handler in
`api/app.py`).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from flask import g

SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sneaker_id TEXT NOT NULL,
    size REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(sneaker_id, size)
);

CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    watchlist_id INTEGER NOT NULL REFERENCES watchlist(id) ON DELETE CASCADE,
    as_of TEXT NOT NULL,
    low REAL NOT NULL,
    mid REAL NOT NULL,
    high REAL NOT NULL,
    extrapolated INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a new connection with foreign-key enforcement turned on.

    SQLite ignores ``ON DELETE CASCADE`` unless ``PRAGMA foreign_keys=ON``
    is set on the *specific connection* running the delete -- it is not a
    database-wide, persisted setting -- so this must run on every new
    connection, not just once in `init_db`.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create the tables if they do not already exist."""
    conn.executescript(SCHEMA)
    conn.commit()


def get_db() -> sqlite3.Connection:
    """Return the request-scoped connection, opening + initializing it lazily."""
    from flask import current_app

    if "db" not in g:
        g.db = connect(current_app.config["DB_PATH"])
        init_db(g.db)
    return g.db


def close_db(_exc: BaseException | None = None) -> None:
    """Close the request-scoped connection, if one was opened."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def add_item(conn: sqlite3.Connection, sneaker_id: str, size: float, created_at: str) -> int:
    """Insert a new watchlist row. Raises `sqlite3.IntegrityError` on a
    duplicate (sneaker_id, size) pair."""
    cur = conn.execute(
        "INSERT INTO watchlist (sneaker_id, size, created_at) VALUES (?, ?, ?)",
        (sneaker_id, size, created_at),
    )
    conn.commit()
    return cur.lastrowid


def list_items(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """All watchlist rows, oldest first."""
    return conn.execute("SELECT * FROM watchlist ORDER BY id ASC").fetchall()


def get_item(conn: sqlite3.Connection, item_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM watchlist WHERE id = ?", (item_id,)).fetchone()


def delete_item(conn: sqlite3.Connection, item_id: int) -> bool:
    """Delete a watchlist row (cascading to its snapshots). Returns whether
    a row was actually deleted."""
    cur = conn.execute("DELETE FROM watchlist WHERE id = ?", (item_id,))
    conn.commit()
    return cur.rowcount > 0


def add_snapshot(
    conn: sqlite3.Connection,
    watchlist_id: int,
    as_of: str,
    low: float,
    mid: float,
    high: float,
    extrapolated: bool,
    created_at: str,
) -> int:
    cur = conn.execute(
        """INSERT INTO snapshots
               (watchlist_id, as_of, low, mid, high, extrapolated, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (watchlist_id, as_of, low, mid, high, int(extrapolated), created_at),
    )
    conn.commit()
    return cur.lastrowid


def snapshots_for(conn: sqlite3.Connection, watchlist_id: int) -> list[sqlite3.Row]:
    """A watchlist item's snapshots, ordered chronologically by `as_of`."""
    return conn.execute(
        "SELECT * FROM snapshots WHERE watchlist_id = ? ORDER BY as_of ASC, id ASC",
        (watchlist_id,),
    ).fetchall()
