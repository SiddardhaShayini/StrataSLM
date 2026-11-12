"""Tests for CoALA 4-tier CognitiveMemoryStore and SleepTimeConsolidator."""

from pathlib import Path

import pytest

from strata.config import StrataConfig
from strata.governor.hardware import MemorySnapshot
from strata.governor.resource_governor import ResourceGovernor
from strata.memory.consolidator import SleepTimeConsolidator, extract_rule_facts
from strata.memory.store import CognitiveMemoryStore
from strata.storage.db import StrataDB


def test_bitemporal_fact_invalidation_hybrid_rrf_and_abstention(tmp_path: Path) -> None:
    with StrataDB(tmp_path / "mem.db") as db:
        mem = CognitiveMemoryStore(db)

        # Working memory bounded queue
        for i in range(5):
            mem.push_working_turn("s1", "user", f"msg {i}", max_turns=3)
        assert [t["content"] for t in mem.get_working_turns("s1")] == ["msg 2", "msg 3", "msg 4"]

        # Bi-temporal knowledge update: Pune (t=100) -> Bengaluru (t=200)
        m1 = mem.upsert_fact("user", "lives_in", "Pune", valid_from=100.0)
        assert m1.op == "ADD"

        # Identical repeat -> NOOP
        m_noop = mem.upsert_fact("user", "lives_in", "Pune", valid_from=150.0)
        assert m_noop.op == "NOOP"

        m2 = mem.upsert_fact("user", "lives_in", "Bengaluru", valid_from=200.0)
        assert m2.op == "UPDATE"
        assert m2.invalidated_fact_ids == [m1.fact_id]

        # Current recall sees only Bengaluru
        current = mem.recall_facts("Where does the user live?")
        assert len(current) == 1
        assert current[0].object_value == "Bengaluru"

        # Point-in-time recall at t=150 sees Pune!
        past = mem.recall_facts("Where does the user live?", as_of=150.0)
        assert len(past) == 1
        assert past[0].object_value == "Pune"

        # Abstention on completely unrelated query
        unrelated = mem.recall_facts("quantum entanglement tensor eigenvalues")
        assert unrelated == []

        # Procedural skill macro cache
        mem.save_skill("inspect_repo", "check repository status and readme", [{"tool": "git_status"}])
        matched = mem.match_skill("Please check repository status and readme file")
        assert matched is not None
        assert matched.name == "inspect_repo"


def test_rule_fact_extraction() -> None:
    facts = extract_rule_facts("Hi, my name is Aarav. I live in Chennai and I prefer concise answers.")
    assert ("user", "name", "Aarav") in facts
    assert ("user", "lives_in", "Chennai") in facts
    assert ("user", "prefers", "concise answers") in facts


@pytest.mark.asyncio
async def test_sleep_consolidator_respects_governor_pressure(tmp_path: Path) -> None:
    class GreenGov(ResourceGovernor):
        def evaluate(self, **kwargs):
            return super().evaluate(
                memory=MemorySnapshot(8192, 3000, 5192, 63.4, 40, 1800, [1]),
                cpu_percent=10.0,
            )

    class YellowGov(ResourceGovernor):
        def evaluate(self, **kwargs):
            return super().evaluate(
                memory=MemorySnapshot(8192, 1500, 6692, 81.7, 40, 1800, [1]),
                cpu_percent=10.0,
            )

    cfg = StrataConfig()
    with StrataDB(tmp_path / "sleep.db") as db:
        mem = CognitiveMemoryStore(db)
        mem.add_episode("s1", "user", "My name is Maya. I live in Pune.", timestamp=100.0)
        mem.add_episode("s1", "user", "I moved to Hyderabad.", timestamp=200.0)

        # Under YELLOW pressure -> consolidator immediately yields without burning CPU/RAM
        paused_rep = await SleepTimeConsolidator(mem, YellowGov(cfg)).run_once()
        assert paused_rep.paused_by_governor is True
        assert paused_rep.episodes_processed == 0

        # Under GREEN headroom -> consolidates both episodes, links facts, and invalidates Pune -> Hyderabad
        ok_rep = await SleepTimeConsolidator(mem, GreenGov(cfg)).run_once()
        assert ok_rep.paused_by_governor is False
        assert ok_rep.episodes_processed == 2
        assert ok_rep.facts_added == 2
        assert ok_rep.facts_updated == 1
        assert ok_rep.links_created == 1

        res = mem.recall_facts("Where does Maya live?")
        assert any(f.object_value == "Hyderabad" for f in res)
        assert all(f.object_value != "Pune" for f in res)
