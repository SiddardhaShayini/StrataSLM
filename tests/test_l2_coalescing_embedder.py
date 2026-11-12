"""Tests for L2SemanticCache, sqlite-vec integration, embedders, and in-flight coalescing."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from strata.config import StrataConfig
from strata.inference.cache import L2SemanticCache
from strata.inference.client import OllamaClient
from strata.inference.engine import InferenceEngine
from strata.memory.embedder import (
    MINILM_ONNX_URL,
    MINILM_TOKENIZER_URL,
    HashingNgramEmbedder,
    create_embedder,
    fetch_minilm_onnx,
)
from strata.storage.db import StrataDB


def test_l2_semantic_cache_vec_and_fallback(tmp_path: Path) -> None:
    embedder = HashingNgramEmbedder(dim=128)
    for enable_vec in (True, False):
        db_file = tmp_path / f"l2_{enable_vec}.db"
        with StrataDB(db_file, enable_vec=enable_vec) as db:
            l2 = L2SemanticCache(db, embedder=embedder, similarity_threshold=0.55)
            l2.put(
                model="qwen3:1.7b",
                think=False,
                prefix_hash="p1",
                schema_hash="s1",
                query="Explain SQLite WAL mode write performance benefits",
                payload={"content": "WAL enables concurrent reads and fast appends."},
            )
            hit = l2.get(
                model="qwen3:1.7b",
                think=False,
                prefix_hash="p1",
                schema_hash="s1",
                query="Explain SQLite WAL mode write performance",
            )
            assert hit.hit is True
            assert hit.tier == "L2_SEMANTIC"
            assert hit.similarity is not None and hit.similarity >= 0.55

            # Different model/prefix scope must NOT hit
            miss_scope = l2.get(
                model="qwen3:0.6b",
                think=False,
                prefix_hash="p1",
                schema_hash="s1",
                query="Explain SQLite WAL mode write performance",
            )
            assert miss_scope.hit is False

            # Unrelated topic must NOT hit
            miss_topic = l2.get(
                model="qwen3:1.7b",
                think=False,
                prefix_hash="p1",
                schema_hash="s1",
                query="Bake sourdough bread recipe temperatures",
            )
            assert miss_topic.hit is False


def test_create_embedder_falls_back_gracefully(tmp_path: Path) -> None:
    emb = create_embedder(prefer_onnx=True, model_dir=tmp_path / "missing")
    assert isinstance(emb, HashingNgramEmbedder)
    vec = emb.encode_one("hello world")
    assert len(vec) == 128


@pytest.mark.asyncio
async def test_inference_engine_coalesces_concurrent_identical_requests(tmp_path: Path) -> None:
    calls = 0

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.08)
        line = json.dumps(
            {
                "model": "qwen3:1.7b",
                "message": {"role": "assistant", "content": "coalesced-answer"},
                "done": True,
                "done_reason": "stop",
            }
        )
        return httpx.Response(200, content=(line + "\n").encode("utf-8"))

    cfg = StrataConfig(data_dir=tmp_path)
    with StrataDB(tmp_path / "coalesce.db") as db:
        async with OllamaClient(transport=httpx.MockTransport(slow_handler)) as client:
            engine = InferenceEngine(cfg, db=db, client=client)
            r1, r2 = await asyncio.gather(
                engine.complete("What is StrataSLM coalescing?"),
                engine.complete("What is StrataSLM coalescing?"),
            )
            assert calls == 1
            assert r1.content == "coalesced-answer"
            assert r2.content == "coalesced-answer"
            assert (r1.coalesced or r2.coalesced) is True


def test_fetch_minilm_onnx_downloads_skips_and_cleans_part(tmp_path: Path) -> None:
    requested: list[str] = []
    fail_next = False

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if fail_next:
            return httpx.Response(503, text="unavailable")
        if str(request.url) == MINILM_ONNX_URL:
            return httpx.Response(200, content=b"mock-onnx-weights-bytes")
        if str(request.url) == MINILM_TOKENIZER_URL:
            return httpx.Response(200, content=b'{"version":"1.0"}')
        return httpx.Response(404)

    target = tmp_path / "minilm"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = fetch_minilm_onnx(target, client=client)
        assert first["downloaded"] == ["model_quantized.onnx", "tokenizer.json"]
        assert first["skipped"] == []
        assert (target / "model_quantized.onnx").read_bytes() == b"mock-onnx-weights-bytes"
        assert (target / "tokenizer.json").read_bytes() == b'{"version":"1.0"}'

        # Second call skips cached files
        second = fetch_minilm_onnx(target, client=client)
        assert second["downloaded"] == []
        assert second["skipped"] == ["model_quantized.onnx", "tokenizer.json"]
        assert len(requested) == 2

        # Force download failure cleans temporary .part files cleanly
        fail_next = True
        with pytest.raises(httpx.HTTPStatusError):
            fetch_minilm_onnx(target, force=True, client=client)
        assert list(target.glob("*.part")) == []
