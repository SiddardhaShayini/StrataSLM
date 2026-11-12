"""Preemptible Idle-Aware Sleep-Time Memory Consolidator (Letta Sleep-Time Compute)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable

from strata.governor.resource_governor import ResourceGovernor
from strata.memory.store import CognitiveMemoryStore

# Rule-first deterministic patterns capture common personal/workspace facts with zero LLM cost!
_FACT_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"\bmy name is\s+([A-Za-z][a-zA-Z0-9_-]{1,30})\b", re.IGNORECASE),
        "user",
        "name",
    ),
    (
        re.compile(
            r"\bi live in\s+([A-Za-z][a-zA-Z0-9 ,_-]{1,40}?)(?:\s+\band\b\s+(?:i|my|we)\b|[.!?,;\n]|$)",
            re.IGNORECASE,
        ),
        "user",
        "lives_in",
    ),
    (
        re.compile(
            r"\bi moved to\s+([A-Za-z][a-zA-Z0-9 ,_-]{1,40}?)(?:\s+\band\b\s+(?:i|my|we)\b|[.!?,;\n]|$)",
            re.IGNORECASE,
        ),
        "user",
        "lives_in",
    ),
    (
        re.compile(r"\bi prefer\s+([^.!?\n]{2,60}?)(?:[.!?\n]|$)", re.IGNORECASE),
        "user",
        "prefers",
    ),
    (
        re.compile(
            r"\bmy favorite\s+([a-zA-Z0-9_ ]{2,25}?)\s+is\s+([^.!?\n]{1,50}?)(?:[.!?\n]|$)",
            re.IGNORECASE,
        ),
        "user",
        "favorite_{group1}",
    ),
)

FactExtractorFn = Callable[[str], Awaitable[list[tuple[str, str, str]]]]


def extract_rule_facts(text: str) -> list[tuple[str, str, str]]:
    """Zero-LLM deterministic extraction of common user/project fact triples."""
    found: list[tuple[str, str, str]] = []
    for pattern, subj, pred in _FACT_PATTERNS:
        for match in pattern.finditer(text):
            if "{group1}" in pred:
                attr = match.group(1).strip().lower().replace(" ", "_")
                val = match.group(2).strip()
                found.append((subj, f"favorite_{attr}", val))
            else:
                val = match.group(1).strip()
                found.append((subj, pred, val))
    return found


@dataclass(slots=True)
class ConsolidationReport:
    episodes_processed: int
    facts_added: int
    facts_updated: int
    facts_noop: int
    links_created: int
    paused_by_governor: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SleepTimeConsolidator:
    """Background memory consolidator that yields immediately under YELLOW/RED pressure."""

    def __init__(
        self,
        store: CognitiveMemoryStore,
        governor: ResourceGovernor,
        *,
        llm_extractor: FactExtractorFn | None = None,
    ) -> None:
        self.store = store
        self.governor = governor
        self.llm_extractor = llm_extractor

    async def run_once(self, *, batch_size: int = 10) -> ConsolidationReport:
        """Process unconsolidated episodes until batch completes or Governor pauses jobs."""
        episodes = self.store.fetch_unconsolidated_episodes(limit=batch_size)
        processed = 0
        added = 0
        updated = 0
        noop = 0
        links = 0
        paused = False

        for ep in episodes:
            assessment = self.governor.evaluate(sample_interval=None)
            if not assessment.allow_sleep_jobs:
                paused = True
                break

            triples = extract_rule_facts(ep["content"])
            if not triples and self.llm_extractor is not None:
                triples = await self.llm_extractor(ep["content"])

            created_ids: list[int] = []
            for subj, pred, obj in triples:
                mut = self.store.upsert_fact(
                    subj,
                    pred,
                    obj,
                    source_episode_id=int(ep["id"]),
                    valid_from=float(ep["created_at"]),
                )
                created_ids.append(mut.fact_id)
                if mut.op == "ADD":
                    added += 1
                elif mut.op == "UPDATE":
                    updated += 1
                else:
                    noop += 1

            # Create A-MEM cross-links between co-occurring facts in the same episode
            for i in range(len(created_ids) - 1):
                self.store.link_facts(created_ids[i], created_ids[i + 1], relation="co_occurring")
                links += 1

            self.store.mark_episode_consolidated(int(ep["id"]))
            processed += 1

        return ConsolidationReport(
            episodes_processed=processed,
            facts_added=added,
            facts_updated=updated,
            facts_noop=noop,
            links_created=links,
            paused_by_governor=paused,
        )
