"""Unit tests for Phase 1 storage, tool pruning, prompt packing, L1 cache, and router."""

from pathlib import Path

from strata.config import StrataConfig
from strata.governor.resource_governor import GovernorState
from strata.inference.cache import L1ExactCache
from strata.inference.prompt import (
    CanonicalPromptBuilder,
    ContextBudget,
    compact_observation,
)
from strata.inference.router import AdaptiveRouter
from strata.inference.tool_pruner import ToolPruner, ToolSpec
from strata.storage.db import StrataDB


def test_tool_pruner_selects_relevant_and_sorts_canonically() -> None:
    tools = [
        ToolSpec("read_file", "Read contents of a local file", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, tags=("fs", "files")),
        ToolSpec("write_file", "Write or overwrite a local file", tags=("fs", "files")),
        ToolSpec("search_sqlite", "Query SQLite FTS5 full-text memory database", tags=("memory", "db")),
        ToolSpec("run_shell", "Execute a sandboxed shell command", tags=("exec",)),
        ToolSpec("git_status", "Show git working tree status", tags=("git",)),
        ToolSpec("get_time", "Get current local timestamp", always_include=True),
    ]
    pruner = ToolPruner(tools)
    res = pruner.select("Please read the local file config.yaml and inspect its contents", top_k=3)

    names = [t.name for t in res.selected]
    assert "get_time" in names  # pinned
    assert "read_file" in names
    assert names == sorted(names)  # canonical alphabetical order for KV prefix reuse
    assert res.selected_count <= 3
    assert res.char_reduction_pct > 0


def test_prompt_packer_prefix_stability_and_budget() -> None:
    tools = [
        ToolSpec("alpha", "First tool"),
        ToolSpec("beta", "Second tool"),
    ]
    builder = CanonicalPromptBuilder(
        "  You are StrataSLM.\n\n\nFollow rules strictly.  ",
        budget=ContextBudget(num_ctx=512, reserved_output_tokens=128),
    )

    long_obs = "LINE " * 400
    compacted, did_compact = compact_observation(long_obs, max_tokens=60)
    assert did_compact is True
    assert "truncated" in compacted

    p1 = builder.pack(
        "Summarize findings",
        tools=tools,
        memory_facts=["User runs Windows AMD64", "RAM limit is 8 GB"],
        history=[
            {"role": "user", "content": "Old message " * 40},
            {"role": "tool", "content": long_obs},
            {"role": "assistant", "content": "Recent reply"},
        ],
    )
    p2 = builder.pack(
        "Different user question",
        tools=list(reversed(tools)),
        memory_facts=["Different memory fact"],
        history=[],
    )

    # Identical static system + tool set yields identical prefix_hash regardless of input tool order!
    assert p1.prefix_hash == p2.prefix_hash
    assert p1.estimated_tokens <= 512 - 128
    assert p1.compacted_observations == 1


def test_l1_exact_cache_mem_sqlite_and_trim(tmp_path: Path) -> None:
    db_path = tmp_path / "strata.db"
    with StrataDB(db_path) as db:
        cache = L1ExactCache(db, max_memory_entries=4, max_db_entries=10)
        miss = cache.get(
            model="qwen3:1.7b",
            think=False,
            prefix_hash="p1",
            schema_hash="s1",
            query="What is StrataSLM?",
        )
        assert miss.hit is False

        cache.put(
            model="qwen3:1.7b",
            think=False,
            prefix_hash="p1",
            schema_hash="s1",
            query="  What is   StrataSLM? ",
            payload={"content": "A local AI runtime."},
        )

        hit_mem = cache.get(
            model="qwen3:1.7b",
            think=False,
            prefix_hash="p1",
            schema_hash="s1",
            query="What is StrataSLM?",
        )
        assert hit_mem.hit is True
        assert hit_mem.tier == "L1_MEM"
        assert hit_mem.payload == {"content": "A local AI runtime."}

        # Trim in-memory LRU (simulating YELLOW governor pressure) -> falls back cleanly to SQLite
        removed = cache.trim_memory(0)
        assert removed == 1

        hit_db = cache.get(
            model="qwen3:1.7b",
            think=False,
            prefix_hash="p1",
            schema_hash="s1",
            query="What is StrataSLM?",
        )
        assert hit_db.hit is True
        assert hit_db.tier == "L1_SQLITE"
        assert hit_db.payload == {"content": "A local AI runtime."}


def test_adaptive_router_decisions() -> None:
    cfg = StrataConfig()
    router = AdaptiveRouter(cfg)

    fast = router.route("Hello there", prefer_fast_model=True)
    assert fast.model == cfg.router_model
    assert fast.think is False
    assert fast.num_ctx == 2048

    deep = router.route(
        "Please design and analyze a multi-step algorithm to refactor the sqlite cache and verify trade-offs step-by-step."
    )
    assert deep.model == cfg.primary_model
    assert deep.think is True
    assert deep.tier == "deep"

    pressured = router.route(
        "Please design and analyze a multi-step algorithm.",
        governor_state=GovernorState.YELLOW,
    )
    assert pressured.think is False
    assert pressured.num_ctx == 2048

    recovery = router.route("fix json", is_recovery=True, governor_state=GovernorState.RED)
    assert recovery.think is True
