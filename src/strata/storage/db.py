"""Unified SQLite WAL database manager with bounded memory footprint."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable


_SCHEMA_V1_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS l1_cache (
    cache_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    think INTEGER NOT NULL,
    prefix_hash TEXT NOT NULL,
    schema_hash TEXT NOT NULL,
    normalized_query TEXT NOT NULL,
    response_json TEXT NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    last_accessed_at REAL NOT NULL,
    expires_at REAL
);

CREATE INDEX IF NOT EXISTS idx_l1_cache_lru
    ON l1_cache(last_accessed_at);
"""

_SCHEMA_V2_SQL = """
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    status TEXT NOT NULL,
    current_state TEXT NOT NULL,
    step_index INTEGER NOT NULL DEFAULT 0,
    state_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES agent_runs(run_id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    phase TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    action_name TEXT,
    input_json TEXT NOT NULL,
    output_json TEXT,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE(run_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_agent_steps_run
    ON agent_steps(run_id, step_index);
"""

_SCHEMA_V3_SQL = """
CREATE TABLE IF NOT EXISTS memory_episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    consolidated INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_episodes_fts USING fts5(
    content,
    content='memory_episodes',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS trg_episodes_ai AFTER INSERT ON memory_episodes BEGIN
    INSERT INTO memory_episodes_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS trg_episodes_ad AFTER DELETE ON memory_episodes BEGIN
    INSERT INTO memory_episodes_fts(memory_episodes_fts, rowid, content) VALUES('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS trg_episodes_au AFTER UPDATE ON memory_episodes BEGIN
    INSERT INTO memory_episodes_fts(memory_episodes_fts, rowid, content) VALUES('delete', old.id, old.content);
    INSERT INTO memory_episodes_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TABLE IF NOT EXISTS memory_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_value TEXT NOT NULL,
    fact_text TEXT NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 1.0,
    embedding_blob BLOB,
    source_episode_id INTEGER,
    created_at REAL NOT NULL,
    valid_from REAL NOT NULL,
    invalid_at REAL,
    superseded_by INTEGER
);

CREATE INDEX IF NOT EXISTS idx_memory_facts_active
    ON memory_facts(subject, predicate, invalid_at);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_facts_fts USING fts5(
    subject,
    predicate,
    object_value,
    fact_text,
    content='memory_facts',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS trg_facts_ai AFTER INSERT ON memory_facts BEGIN
    INSERT INTO memory_facts_fts(rowid, subject, predicate, object_value, fact_text)
    VALUES (new.id, new.subject, new.predicate, new.object_value, new.fact_text);
END;
CREATE TRIGGER IF NOT EXISTS trg_facts_ad AFTER DELETE ON memory_facts BEGIN
    INSERT INTO memory_facts_fts(memory_facts_fts, rowid, subject, predicate, object_value, fact_text)
    VALUES('delete', old.id, old.subject, old.predicate, old.object_value, old.fact_text);
END;
CREATE TRIGGER IF NOT EXISTS trg_facts_au AFTER UPDATE ON memory_facts BEGIN
    INSERT INTO memory_facts_fts(memory_facts_fts, rowid, subject, predicate, object_value, fact_text)
    VALUES('delete', old.id, old.subject, old.predicate, old.object_value, old.fact_text);
    INSERT INTO memory_facts_fts(rowid, subject, predicate, object_value, fact_text)
    VALUES (new.id, new.subject, new.predicate, new.object_value, new.fact_text);
END;

CREATE TABLE IF NOT EXISTS memory_links (
    src_fact_id INTEGER NOT NULL REFERENCES memory_facts(id) ON DELETE CASCADE,
    dst_fact_id INTEGER NOT NULL REFERENCES memory_facts(id) ON DELETE CASCADE,
    relation TEXT NOT NULL DEFAULT 'related',
    weight REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    PRIMARY KEY (src_fact_id, dst_fact_id, relation)
);

CREATE TABLE IF NOT EXISTS memory_skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    goal_pattern TEXT NOT NULL,
    steps_json TEXT NOT NULL,
    success_count INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    last_used_at REAL NOT NULL
);
"""

_SCHEMA_V4_SQL = """
CREATE TABLE IF NOT EXISTS l2_cache (
    cache_key TEXT PRIMARY KEY,
    scope_key TEXT NOT NULL,
    model TEXT NOT NULL,
    think INTEGER NOT NULL,
    prefix_hash TEXT NOT NULL,
    schema_hash TEXT NOT NULL,
    context_hash TEXT NOT NULL,
    normalized_query TEXT NOT NULL,
    embedding_blob BLOB NOT NULL,
    response_json TEXT NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    last_accessed_at REAL NOT NULL,
    expires_at REAL
);

CREATE INDEX IF NOT EXISTS idx_l2_cache_scope
    ON l2_cache(scope_key, last_accessed_at);
"""


def _try_load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Attempt to load sqlite-vec extension cleanly without hard failing."""
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)
        return True
    except Exception:
        return False


def connect_sqlite(
    db_path: Path | str, *, cache_kb: int = 8000, enable_vec: bool = True
) -> tuple[sqlite3.Connection, bool]:
    """Open a SQLite connection configured for WAL mode and 8 GB RAM limits."""
    path_str = str(db_path)
    if path_str != ":memory:":
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path_str, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute(f"PRAGMA cache_size=-{abs(int(cache_kb))};")
    has_vec = _try_load_sqlite_vec(conn) if enable_vec else False
    return conn, has_vec


class StrataDB:
    """Single-file SQLite WAL store shared across cache, checkpoints, and memory."""

    def __init__(
        self, db_path: Path | str = Path(".strata/strata.db"), *, enable_vec: bool = True
    ) -> None:
        self.db_path = Path(db_path) if str(db_path) != ":memory:" else ":memory:"
        self.conn, self.has_vec = connect_sqlite(self.db_path, enable_vec=enable_vec)
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(_SCHEMA_V1_SQL)
            if (
                self.conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = 1"
                ).fetchone()
                is None
            ):
                self.conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES (1)"
                )

            self.conn.executescript(_SCHEMA_V2_SQL)
            if (
                self.conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = 2"
                ).fetchone()
                is None
            ):
                self.conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES (2)"
                )

            self.conn.executescript(_SCHEMA_V3_SQL)
            if (
                self.conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = 3"
                ).fetchone()
                is None
            ):
                self.conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES (3)"
                )

            self.conn.executescript(_SCHEMA_V4_SQL)
            if (
                self.conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = 4"
                ).fetchone()
                is None
            ):
                self.conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES (4)"
                )

    def execute(self, sql: str, params: Iterable[object] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, tuple(params))

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> StrataDB:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
