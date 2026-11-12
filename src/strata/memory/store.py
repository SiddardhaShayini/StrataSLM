"""CoALA 4-tier SQLite memory store with bi-temporal facts and Hybrid RRF retrieval."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from strata.memory.embedder import (
    HashingNgramEmbedder,
    TextEmbedder,
    cosine_similarity,
    pack_vector,
    unpack_vector,
)
from strata.storage.db import StrataDB

_WORD_RE = re.compile(r"[a-z0-9]+")


_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
        "for", "from", "has", "have", "he", "her", "his", "how", "i", "in",
        "is", "it", "its", "me", "my", "of", "on", "or", "our", "she", "that",
        "the", "their", "them", "they", "this", "to", "was", "we", "were",
        "what", "when", "where", "which", "who", "why", "with", "you", "your",
    }
)


def _build_fts_or_query(text: str) -> str | None:
    tokens = []
    seen: set[str] = set()
    for tok in _WORD_RE.findall(text.lower()):
        if len(tok) >= 2 and tok not in _STOP_WORDS and tok not in seen:
            seen.add(tok)
            tokens.append(tok)
    if not tokens:
        return None
    return " OR ".join(tokens[:12])


@dataclass(slots=True)
class SemanticFact:
    id: int
    subject: str
    predicate: str
    object_value: str
    fact_text: str
    tags: list[str]
    confidence: float
    created_at: float
    valid_from: float
    invalid_at: float | None
    superseded_by: int | None
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemoryMutation:
    op: str  # "ADD" | "UPDATE" | "NOOP"
    fact_id: int
    invalidated_fact_ids: list[int]


@dataclass(slots=True)
class SkillMacro:
    id: int
    name: str
    goal_pattern: str
    steps: list[dict[str, Any]]
    success_count: int
    last_used_at: float


class CognitiveMemoryStore:
    """Unified Working / Episodic / Bi-Temporal Semantic / Procedural memory in strata.db."""

    def __init__(
        self,
        db: StrataDB,
        *,
        embedder: TextEmbedder | None = None,
        min_semantic_similarity: float = 0.20,
    ) -> None:
        self.db = db
        self.embedder: TextEmbedder = embedder or HashingNgramEmbedder(dim=128)
        self.min_semantic_similarity = min_semantic_similarity
        self._working: dict[str, list[dict[str, str]]] = {}

    # --- 1. Working Memory (bounded in-RAM per session) ---
    def push_working_turn(
        self, session_id: str, role: str, content: str, *, max_turns: int = 12
    ) -> None:
        turns = self._working.setdefault(session_id, [])
        turns.append({"role": role, "content": content})
        if len(turns) > max_turns:
            del turns[: len(turns) - max_turns]

    def get_working_turns(self, session_id: str) -> list[dict[str, str]]:
        return list(self._working.get(session_id, ()))

    def clear_working(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._working.clear()
        else:
            self._working.pop(session_id, None)

    # --- 2. Episodic Memory ---
    def add_episode(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        importance: float = 0.5,
        timestamp: float | None = None,
    ) -> int:
        ts = time.time() if timestamp is None else float(timestamp)
        with self.db.conn:
            cur = self.db.execute(
                """
                INSERT INTO memory_episodes (session_id, role, content, importance, consolidated, created_at)
                VALUES (?, ?, ?, ?, 0, ?)
                """,
                (session_id, role, content, float(importance), ts),
            )
            return int(cur.lastrowid)

    def fetch_unconsolidated_episodes(self, limit: int = 25) -> list[dict[str, Any]]:
        rows = self.db.execute(
            """
            SELECT id, session_id, role, content, importance, created_at
            FROM memory_episodes
            WHERE consolidated = 0
            ORDER BY id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_episode_consolidated(self, episode_id: int) -> None:
        with self.db.conn:
            self.db.execute(
                "UPDATE memory_episodes SET consolidated = 1 WHERE id = ?",
                (episode_id,),
            )

    # --- 3. Bi-Temporal Semantic Graph ---
    def upsert_fact(
        self,
        subject: str,
        predicate: str,
        object_value: str,
        *,
        fact_text: str | None = None,
        tags: Sequence[str] = (),
        confidence: float = 1.0,
        source_episode_id: int | None = None,
        valid_from: float | None = None,
    ) -> MemoryMutation:
        """Insert or supersede active fact for `(subject, predicate)` cleanly (Mem0 + Zep)."""
        norm_subj = subject.strip().lower()
        norm_pred = predicate.strip().lower()
        norm_obj = object_value.strip()
        text = fact_text or f"{subject.strip()} {predicate.strip().replace('_', ' ')} {norm_obj}"
        now = time.time()
        eff_from = now if valid_from is None else float(valid_from)
        vec_blob = pack_vector(self.embedder.encode_one(text))

        active_rows = self.db.execute(
            """
            SELECT id, object_value FROM memory_facts
            WHERE subject = ? AND predicate = ? AND invalid_at IS NULL
            ORDER BY id DESC
            """,
            (norm_subj, norm_pred),
        ).fetchall()

        for row in active_rows:
            if row["object_value"].strip().lower() == norm_obj.lower():
                return MemoryMutation(
                    op="NOOP",
                    fact_id=int(row["id"]),
                    invalidated_fact_ids=[],
                )

        with self.db.conn:
            cur = self.db.execute(
                """
                INSERT INTO memory_facts (
                    subject, predicate, object_value, fact_text, tags_json,
                    confidence, embedding_blob, source_episode_id,
                    created_at, valid_from, invalid_at, superseded_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    norm_subj,
                    norm_pred,
                    norm_obj,
                    text,
                    json.dumps(list(tags)),
                    float(confidence),
                    vec_blob,
                    source_episode_id,
                    now,
                    eff_from,
                ),
            )
            new_id = int(cur.lastrowid)

            invalidated: list[int] = []
            for row in active_rows:
                old_id = int(row["id"])
                self.db.execute(
                    """
                    UPDATE memory_facts
                    SET invalid_at = ?, superseded_by = ?
                    WHERE id = ?
                    """,
                    (eff_from, new_id, old_id),
                )
                invalidated.append(old_id)

        op = "UPDATE" if invalidated else "ADD"
        return MemoryMutation(op=op, fact_id=new_id, invalidated_fact_ids=invalidated)

    def link_facts(
        self, src_fact_id: int, dst_fact_id: int, *, relation: str = "related", weight: float = 1.0
    ) -> None:
        if src_fact_id == dst_fact_id:
            return
        now = time.time()
        with self.db.conn:
            self.db.execute(
                """
                INSERT INTO memory_links (src_fact_id, dst_fact_id, relation, weight, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(src_fact_id, dst_fact_id, relation) DO UPDATE SET
                    weight = excluded.weight
                """,
                (src_fact_id, dst_fact_id, relation, float(weight), now),
            )

    def recall_facts(
        self,
        query: str,
        *,
        top_k: int = 5,
        as_of: float | None = None,
        rrf_k: int = 60,
    ) -> list[SemanticFact]:
        """Hybrid RRF (FTS5 BM25 + vector cosine) with bi-temporal filter & abstention."""
        if not query.strip():
            return []

        if as_of is None:
            temporal_where = "f.invalid_at IS NULL"
            temporal_params: tuple[Any, ...] = ()
        else:
            temporal_where = "f.valid_from <= ? AND (f.invalid_at IS NULL OR f.invalid_at > ?)"
            temporal_params = (float(as_of), float(as_of))

        # 1. Lexical FTS5 BM25 ranking
        fts_ranks: dict[int, int] = {}
        fts_query = _build_fts_or_query(query)
        if fts_query:
            fts_rows = self.db.execute(
                f"""
                SELECT f.id
                FROM memory_facts_fts fts
                JOIN memory_facts f ON f.id = fts.rowid
                WHERE memory_facts_fts MATCH ? AND {temporal_where}
                ORDER BY bm25(memory_facts_fts) ASC
                LIMIT ?
                """,
                (fts_query, *temporal_params, max(top_k * 4, 20)),
            ).fetchall()
            for rank, row in enumerate(fts_rows, start=1):
                fts_ranks[int(row["id"])] = rank

        # 2. Vector similarity ranking over active/as-of facts (uses sqlite-vec when loaded)
        q_vec = self.embedder.encode_one(query)
        q_blob = pack_vector(q_vec)
        q_blob_len = len(q_blob)

        if self.db.has_vec:
            cand_rows = self.db.execute(
                f"""
                SELECT f.*,
                       CASE
                           WHEN f.embedding_blob IS NOT NULL AND length(f.embedding_blob) = ?
                           THEN 1.0 - vec_distance_cosine(f.embedding_blob, ?)
                           ELSE 0.0
                       END AS vec_sim
                FROM memory_facts f
                WHERE {temporal_where}
                ORDER BY vec_sim DESC, f.id DESC
                LIMIT 1000
                """,
                (q_blob_len, q_blob, *temporal_params),
            ).fetchall()
        else:
            cand_rows = self.db.execute(
                f"""
                SELECT * FROM memory_facts f
                WHERE {temporal_where}
                ORDER BY f.id DESC
                LIMIT 1000
                """,
                temporal_params,
            ).fetchall()

        if not cand_rows:
            return []

        vec_scored: list[tuple[float, dict[str, Any]]] = []
        row_by_id: dict[int, dict[str, Any]] = {}
        for row in cand_rows:
            d = dict(row)
            fid = int(d["id"])
            row_by_id[fid] = d
            if "vec_sim" in d:
                sim = float(d["vec_sim"]) if d["vec_sim"] is not None else 0.0
            else:
                sim = cosine_similarity(q_vec, unpack_vector(d.get("embedding_blob")))
            vec_scored.append((sim, d))

        vec_scored.sort(key=lambda item: (-item[0], -int(item[1]["id"])))
        vec_ranks: dict[int, int] = {}
        vec_sims: dict[int, float] = {}
        for rank, (sim, d) in enumerate(vec_scored, start=1):
            fid = int(d["id"])
            vec_sims[fid] = sim
            if sim >= self.min_semantic_similarity:
                vec_ranks[fid] = rank

        # Abstention guard: if neither lexical nor semantic threshold matched, return []
        candidate_ids = set(fts_ranks.keys()) | set(vec_ranks.keys())
        if not candidate_ids:
            return []

        fused_map: dict[int, float] = {}
        for fid in candidate_ids:
            score = 0.0
            if fid in fts_ranks:
                score += 1.0 / (rrf_k + fts_ranks[fid])
            if fid in vec_ranks:
                score += 1.0 / (rrf_k + vec_ranks[fid])
            score += max(0.0, vec_sims.get(fid, 0.0)) * 0.01
            fused_map[fid] = score

        # 3. 1-hop A-MEM graph expansion from top direct hits (surfacing linked active facts)
        top_seeds = sorted(fused_map.items(), key=lambda kv: -kv[1])[:3]
        for seed_id, seed_score in top_seeds:
            neighbors = self.db.execute(
                """
                SELECT dst_fact_id AS nid FROM memory_links WHERE src_fact_id = ?
                UNION
                SELECT src_fact_id AS nid FROM memory_links WHERE dst_fact_id = ?
                """,
                (seed_id, seed_id),
            ).fetchall()
            for nb in neighbors:
                nid = int(nb["nid"])
                if nid in row_by_id:
                    boost = seed_score * 0.5
                    if nid not in fused_map or boost > fused_map[nid]:
                        fused_map[nid] = max(fused_map.get(nid, 0.0), boost)

        fused = sorted(((sc, fid) for fid, sc in fused_map.items()), key=lambda item: (-item[0], -item[1]))

        results: list[SemanticFact] = []
        for score, fid in fused[:top_k]:
            d = row_by_id[fid]
            results.append(
                SemanticFact(
                    id=fid,
                    subject=d["subject"],
                    predicate=d["predicate"],
                    object_value=d["object_value"],
                    fact_text=d["fact_text"],
                    tags=json.loads(d["tags_json"]),
                    confidence=float(d["confidence"]),
                    created_at=float(d["created_at"]),
                    valid_from=float(d["valid_from"]),
                    invalid_at=float(d["invalid_at"]) if d["invalid_at"] is not None else None,
                    superseded_by=int(d["superseded_by"]) if d["superseded_by"] is not None else None,
                    score=round(score, 6),
                )
            )
        return results

    # --- 4. Procedural Skill Macro Cache ---
    def save_skill(
        self, name: str, goal_pattern: str, steps: Sequence[dict[str, Any]]
    ) -> int:
        now = time.time()
        serialized = json.dumps(list(steps))
        with self.db.conn:
            self.db.execute(
                """
                INSERT INTO memory_skills (name, goal_pattern, steps_json, success_count, created_at, last_used_at)
                VALUES (?, ?, ?, 1, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    goal_pattern = excluded.goal_pattern,
                    steps_json = excluded.steps_json,
                    success_count = memory_skills.success_count + 1,
                    last_used_at = excluded.last_used_at
                """,
                (name, goal_pattern, serialized, now, now),
            )
            row = self.db.execute(
                "SELECT id FROM memory_skills WHERE name = ?", (name,)
            ).fetchone()
            return int(row["id"])

    def match_skill(self, goal: str) -> SkillMacro | None:
        q_tokens = set(_WORD_RE.findall(goal.lower()))
        if not q_tokens:
            return None
        rows = self.db.execute(
            "SELECT * FROM memory_skills ORDER BY success_count DESC, last_used_at DESC"
        ).fetchall()
        best: tuple[float, SkillMacro] | None = None
        for row in rows:
            pat_tokens = set(_WORD_RE.findall(row["goal_pattern"].lower()))
            if not pat_tokens:
                continue
            overlap = len(q_tokens & pat_tokens) / len(pat_tokens)
            if overlap >= 0.5 and (best is None or overlap > best[0]):
                best = (
                    overlap,
                    SkillMacro(
                        id=int(row["id"]),
                        name=row["name"],
                        goal_pattern=row["goal_pattern"],
                        steps=json.loads(row["steps_json"]),
                        success_count=int(row["success_count"]),
                        last_used_at=float(row["last_used_at"]),
                    ),
                )
        return best[1] if best is not None else None
