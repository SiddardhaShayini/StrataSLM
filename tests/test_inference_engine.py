"""Tests for coordinated InferenceEngine (cache hits, routing, RED shedding)."""

import json
from pathlib import Path

import httpx
import pytest

from strata.config import StrataConfig
from strata.governor.hardware import MemorySnapshot
from strata.governor.resource_governor import ResourceGovernor
from strata.inference.client import OllamaClient
from strata.inference.engine import InferenceEngine, ResourcePressureError
from strata.inference.tool_pruner import ToolSpec
from strata.storage.db import StrataDB


@pytest.mark.asyncio
async def test_inference_engine_caches_and_isolates_history(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        line = json.dumps(
            {
                "model": "qwen3:1.7b",
                "message": {"role": "assistant", "content": f"reply-{calls}"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 15,
                "prompt_eval_duration": 100_000_000,
                "eval_count": 5,
                "eval_duration": 100_000_000,
            }
        )
        return httpx.Response(200, content=(line + "\n").encode("utf-8"))

    cfg = StrataConfig(data_dir=tmp_path)
    tools = [
        ToolSpec("read_file", "Read a file"),
        ToolSpec("write_file", "Write a file"),
    ]
    with StrataDB(tmp_path / "strata.db") as db:
        async with OllamaClient(transport=httpx.MockTransport(handler)) as client:
            engine = InferenceEngine(cfg, db=db, client=client)

            r1 = await engine.complete("Read README.md", tools=tools)
            assert r1.cached is False
            assert r1.content == "reply-1"
            assert calls == 1

            # Same query + whitespace variation -> sub-ms L1 hit without hitting Ollama!
            r2 = await engine.complete("  Read   README.md ", tools=tools)
            assert r2.cached is True
            assert r2.cache_tier == "L1_MEM"
            assert r2.content == "reply-1"
            assert calls == 1

            # Different history must NOT falsely hit previous L1 entry!
            r3 = await engine.complete(
                "Read README.md",
                tools=tools,
                history=[{"role": "user", "content": "prior turn"}],
            )
            assert r3.cached is False
            assert r3.content == "reply-2"
            assert calls == 2


@pytest.mark.asyncio
async def test_inference_engine_sheds_uncached_on_red_pressure(tmp_path: Path) -> None:
    class RedGovernor(ResourceGovernor):
        def evaluate(self, **kwargs):
            low_mem = MemorySnapshot(
                total_mb=8192.0,
                available_mb=900.0,
                used_mb=7292.0,
                percent_used=89.0,
                strata_rss_mb=50.0,
                ollama_rss_mb=1800.0,
                ollama_pids=[1],
            )
            return super().evaluate(memory=low_mem, cpu_percent=40.0)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Ollama should not be called under RED pressure")

    from strata.inference.engine import DEFAULT_SYSTEM_PROMPT
    from strata.inference.prompt import CanonicalPromptBuilder, ContextBudget

    cfg = StrataConfig(data_dir=tmp_path)
    with StrataDB(tmp_path / "strata.db") as db:
        async with OllamaClient(transport=httpx.MockTransport(handler)) as client:
            engine = InferenceEngine(cfg, db=db, client=client, governor=RedGovernor(cfg))
            # Pre-populate cache -> still serves cached hits even under RED!
            packed = CanonicalPromptBuilder(
                DEFAULT_SYSTEM_PROMPT,
                budget=ContextBudget(num_ctx=2048, reserved_output_tokens=256),
            ).pack("cached q")
            engine.cache.put(
                model=cfg.primary_model,
                think=False,
                prefix_hash=packed.prefix_hash,
                schema_hash="no_tools",
                query="cached q",
                payload={"content": "ok"},
            )
            hit = await engine.complete("cached q")
            assert hit.cached is True
            assert hit.content == "ok"

            with pytest.raises(ResourcePressureError):
                await engine.complete("uncached query")
