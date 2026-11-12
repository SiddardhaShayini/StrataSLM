"""L1 exact-match normalized inference cache (in-memory LRU + SQLite WAL)."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any

from strata.memory.embedder import (
    HashingNgramEmbedder,
    TextEmbedder,
    cosine_similarity,
    pack_vector,
    unpack_vector,
)
from strata.storage.db import StrataDB

_SPACE_RE = re.compile(r"\s+")


def normalize_cache_text(text: str) -> str:
    """Normalize prompt text for deterministic exact cache matching."""
    return _SPACE_RE.sub(" ", text.strip())


def build_l1_key(
    *,
    model: str,
    think: bool,
    prefix_hash: str,
    schema_hash: str,
    query: str,
    context_hash: str = "",
) -> tuple[str, str]:
    normalized = normalize_cache_text(query)
    raw = json.dumps(
        [model, bool(think), prefix_hash, schema_hash, context_hash, normalized],
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest(), normalized


@dataclass(slots=True)
class CacheLookupResult:
    hit: bool
    tier: str | None
    payload: dict[str, Any] | None
    latency_ms: float
    similarity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class L1ExactCache:
    """Two-level L1 cache: sub-ms bounded RAM LRU backed by SQLite WAL `l1_cache`."""

    def __init__(
        self,
        db: StrataDB,
        *,
        max_memory_entries: int = 256,
        max_db_entries: int = 5000,
        default_ttl_sec: float = 86400.0,
    ) -> None:
        self._db = db
        self._max_mem = max_memory_entries
        self._max_db = max_db_entries
        self._default_ttl = default_ttl_sec
        self._lru: OrderedDict[str, tuple[dict[str, Any], float | None]] = OrderedDict()

    def get(
        self,
        *,
        model: str,
        think: bool,
        prefix_hash: str,
        schema_hash: str,
        query: str,
        context_hash: str = "",
    ) -> CacheLookupResult:
        t0 = time.perf_counter()
        key, _ = build_l1_key(
            model=model,
            think=think,
            prefix_hash=prefix_hash,
            schema_hash=schema_hash,
            query=query,
            context_hash=context_hash,
        )
        now = time.time()

        # 1. Fast in-memory LRU check
        mem_entry = self._lru.get(key)
        if mem_entry is not None:
            payload, expires_at = mem_entry
            if expires_at is None or expires_at > now:
                self._lru.move_to_end(key)
                return CacheLookupResult(
                    hit=True,
                    tier="L1_MEM",
                    payload=payload,
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                )
            del self._lru[key]

        # 2. SQLite WAL fallback
        row = self._db.execute(
            "SELECT response_json, expires_at FROM l1_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row is not None:
            expires_at = row["expires_at"]
            if expires_at is None or float(expires_at) > now:
                payload = json.loads(row["response_json"])
                with self._db.conn:
                    self._db.execute(
                        "UPDATE l1_cache SET hit_count = hit_count + 1, last_accessed_at = ? WHERE cache_key = ?",
                        (now, key),
                    )
                self._remember_in_lru(key, payload, expires_at)
                return CacheLookupResult(
                    hit=True,
                    tier="L1_SQLITE",
                    payload=payload,
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                )
            with self._db.conn:
                self._db.execute("DELETE FROM l1_cache WHERE cache_key = ?", (key,))

        return CacheLookupResult(
            hit=False,
            tier=None,
            payload=None,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
        )

    def put(
        self,
        *,
        model: str,
        think: bool,
        prefix_hash: str,
        schema_hash: str,
        query: str,
        payload: dict[str, Any],
        context_hash: str = "",
        ttl_sec: float | None = None,
    ) -> str:
        key, normalized = build_l1_key(
            model=model,
            think=think,
            prefix_hash=prefix_hash,
            schema_hash=schema_hash,
            query=query,
            context_hash=context_hash,
        )
        now = time.time()
        effective_ttl = self._default_ttl if ttl_sec is None else ttl_sec
        expires_at = now + effective_ttl if effective_ttl > 0 else None
        serialized = json.dumps(payload, separators=(",", ":"))

        with self._db.conn:
            self._db.execute(
                """
                INSERT INTO l1_cache (
                    cache_key, model, think, prefix_hash, schema_hash,
                    normalized_query, response_json, hit_count,
                    created_at, last_accessed_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    response_json = excluded.response_json,
                    last_accessed_at = excluded.last_accessed_at,
                    expires_at = excluded.expires_at
                """,
                (
                    key,
                    model,
                    int(bool(think)),
                    prefix_hash,
                    schema_hash,
                    normalized,
                    serialized,
                    now,
                    now,
                    expires_at,
                ),
            )
            self._evict_db_overflow()

        self._remember_in_lru(key, payload, expires_at)
        return key

    def trim_memory(self, max_entries: int = 0) -> int:
        """Trim in-memory LRU when Resource Governor enters YELLOW/RED pressure."""
        removed = 0
        target = max(0, max_entries)
        while len(self._lru) > target:
            self._lru.popitem(last=False)
            removed += 1
        return removed

    def _remember_in_lru(
        self, key: str, payload: dict[str, Any], expires_at: float | None
    ) -> None:
        if self._max_mem <= 0:
            return
        self._lru[key] = (payload, expires_at)
        self._lru.move_to_end(key)
        while len(self._lru) > self._max_mem:
            self._lru.popitem(last=False)

    def _evict_db_overflow(self) -> None:
        count = int(self._db.execute("SELECT COUNT(*) FROM l1_cache").fetchone()[0])
        excess = count - self._max_db
        if excess > 0:
            self._db.execute(
                """
                DELETE FROM l1_cache WHERE cache_key IN (
                    SELECT cache_key FROM l1_cache
                    ORDER BY last_accessed_at ASC
                    LIMIT ?
                )
                """,
                (excess,),
            )


def build_l2_scope_key(
    *,
    model: str,
    think: bool,
    prefix_hash: str,
    schema_hash: str,
    context_hash: str = "",
) -> str:
    raw = json.dumps(
        [model, bool(think), prefix_hash, schema_hash, context_hash],
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class L2SemanticCache:
    """SQLite-backed semantic inference cache using sqlite-vec or pure-Python cosine."""

    def __init__(
        self,
        db: StrataDB,
        *,
        embedder: TextEmbedder | None = None,
        similarity_threshold: float = 0.88,
        max_db_entries: int = 2000,
        default_ttl_sec: float = 86400.0,
    ) -> None:
        self._db = db
        self.embedder: TextEmbedder = embedder or HashingNgramEmbedder(dim=128)
        self.similarity_threshold = similarity_threshold
        self._max_db = max_db_entries
        self._default_ttl = default_ttl_sec

    def get(
        self,
        *,
        model: str,
        think: bool,
        prefix_hash: str,
        schema_hash: str,
        query: str,
        context_hash: str = "",
        threshold: float | None = None,
    ) -> CacheLookupResult:
        t0 = time.perf_counter()
        normalized = normalize_cache_text(query)
        if not normalized:
            return CacheLookupResult(hit=False, tier=None, payload=None, latency_ms=0.0)

        scope_key = build_l2_scope_key(
            model=model,
            think=think,
            prefix_hash=prefix_hash,
            schema_hash=schema_hash,
            context_hash=context_hash,
        )
        q_vec = self.embedder.encode_one(normalized)
        q_blob = pack_vector(q_vec)
        q_blob_len = len(q_blob)
        now = time.time()
        min_sim = self.similarity_threshold if threshold is None else threshold

        best_key: str | None = None
        best_sim = -1.0
        best_payload_raw: str | None = None

        if self._db.has_vec:
            row = self._db.execute(
                """
                SELECT cache_key, response_json,
                       1.0 - vec_distance_cosine(embedding_blob, ?) AS sim
                FROM l2_cache
                WHERE scope_key = ?
                  AND length(embedding_blob) = ?
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY sim DESC
                LIMIT 1
                """,
                (q_blob, scope_key, q_blob_len, now),
            ).fetchone()
            if row is not None and row["sim"] is not None:
                best_key = row["cache_key"]
                best_sim = float(row["sim"])
                best_payload_raw = row["response_json"]
        else:
            rows = self._db.execute(
                """
                SELECT cache_key, embedding_blob, response_json
                FROM l2_cache
                WHERE scope_key = ?
                  AND length(embedding_blob) = ?
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY last_accessed_at DESC
                LIMIT 250
                """,
                (scope_key, q_blob_len, now),
            ).fetchall()
            for r in rows:
                sim = cosine_similarity(q_vec, unpack_vector(r["embedding_blob"]))
                if sim > best_sim:
                    best_sim = sim
                    best_key = r["cache_key"]
                    best_payload_raw = r["response_json"]

        if best_key is not None and best_payload_raw is not None and best_sim >= min_sim:
            with self._db.conn:
                self._db.execute(
                    "UPDATE l2_cache SET hit_count = hit_count + 1, last_accessed_at = ? WHERE cache_key = ?",
                    (now, best_key),
                )
            return CacheLookupResult(
                hit=True,
                tier="L2_SEMANTIC",
                payload=json.loads(best_payload_raw),
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                similarity=round(best_sim, 4),
            )

        return CacheLookupResult(
            hit=False,
            tier=None,
            payload=None,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            similarity=round(best_sim, 4) if best_sim >= 0.0 else None,
        )

    def put(
        self,
        *,
        model: str,
        think: bool,
        prefix_hash: str,
        schema_hash: str,
        query: str,
        payload: dict[str, Any],
        context_hash: str = "",
        ttl_sec: float | None = None,
    ) -> str:
        cache_key, normalized = build_l1_key(
            model=model,
            think=think,
            prefix_hash=prefix_hash,
            schema_hash=schema_hash,
            query=query,
            context_hash=context_hash,
        )
        scope_key = build_l2_scope_key(
            model=model,
            think=think,
            prefix_hash=prefix_hash,
            schema_hash=schema_hash,
            context_hash=context_hash,
        )
        vec_blob = pack_vector(self.embedder.encode_one(normalized))
        now = time.time()
        effective_ttl = self._default_ttl if ttl_sec is None else ttl_sec
        expires_at = now + effective_ttl if effective_ttl > 0 else None
        serialized = json.dumps(payload, separators=(",", ":"))

        with self._db.conn:
            self._db.execute(
                """
                INSERT INTO l2_cache (
                    cache_key, scope_key, model, think, prefix_hash, schema_hash,
                    context_hash, normalized_query, embedding_blob, response_json,
                    hit_count, created_at, last_accessed_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    embedding_blob = excluded.embedding_blob,
                    response_json = excluded.response_json,
                    last_accessed_at = excluded.last_accessed_at,
                    expires_at = excluded.expires_at
                """,
                (
                    cache_key,
                    scope_key,
                    model,
                    int(bool(think)),
                    prefix_hash,
                    schema_hash,
                    context_hash,
                    normalized,
                    vec_blob,
                    serialized,
                    now,
                    now,
                    expires_at,
                ),
            )
            count = int(self._db.execute("SELECT COUNT(*) FROM l2_cache").fetchone()[0])
            excess = count - self._max_db
            if excess > 0:
                self._db.execute(
                    """
                    DELETE FROM l2_cache WHERE cache_key IN (
                        SELECT cache_key FROM l2_cache
                        ORDER BY last_accessed_at ASC
                        LIMIT ?
                    )
                    """,
                    (excess,),
                )
        return cache_key
