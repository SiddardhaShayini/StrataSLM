"""Tests for async OllamaClient using httpx.MockTransport."""

import asyncio
import json

import httpx
import pytest

from strata.inference.client import OllamaClient, RequestPriority


@pytest.mark.asyncio
async def test_ollama_health_and_streaming_chat() -> None:
    captured_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.16"})
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "qwen3:1.7b",
                            "size": 1_400_000_000,
                            "details": {"parameter_size": "1.7B", "quantization_level": "Q4_K_M"},
                        }
                    ]
                },
            )
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/chat":
            body = json.loads(request.content.decode("utf-8"))
            captured_payloads.append(body)
            lines = [
                json.dumps({"model": "qwen3:1.7b", "message": {"role": "assistant", "thinking": "plan "}, "done": False}),
                json.dumps({"model": "qwen3:1.7b", "message": {"role": "assistant", "content": "Hello"}, "done": False}),
                json.dumps(
                    {
                        "model": "qwen3:1.7b",
                        "message": {"role": "assistant", "content": " world"},
                        "done": True,
                        "done_reason": "stop",
                        "load_duration": 10_000_000,
                        "prompt_eval_count": 20,
                        "prompt_eval_duration": 200_000_000,
                        "eval_count": 10,
                        "eval_duration": 250_000_000,
                    }
                ),
            ]
            return httpx.Response(200, content="\n".join(lines).encode("utf-8"))
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with OllamaClient(transport=transport) as client:
        health = await client.health()
        assert health.reachable is True
        assert health.version == "0.40.16"
        assert health.has_model("qwen3:1.7b") is True
        assert health.has_model("qwen3:0.6b") is False

        res = await client.chat(
            "qwen3:1.7b",
            [{"role": "user", "content": "hi"}],
            think=True,
            num_ctx=4096,
            num_thread=4,
            stream=True,
        )
        assert res.content == "Hello world"
        assert res.thinking == "plan "
        assert res.done_reason == "stop"
        assert res.metrics.prompt_eval_count == 20
        assert res.metrics.eval_count == 10
        assert res.metrics.prompt_tok_per_sec == 100.0
        assert res.metrics.eval_tok_per_sec == 40.0
        assert res.metrics.first_content_ms is not None
        assert captured_payloads[0]["think"] is True
        assert captured_payloads[0]["options"]["num_thread"] == 4

        await client.unload_model("qwen3:1.7b")


@pytest.mark.asyncio
async def test_ollama_priority_gate_preempts_background() -> None:
    order: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        tag = body["messages"][-1]["content"]
        order.append(tag)
        if tag == "hold":
            await asyncio.sleep(0.05)
        line = json.dumps(
            {
                "model": "qwen3:1.7b",
                "message": {"role": "assistant", "content": tag},
                "done": True,
            }
        )
        return httpx.Response(200, content=(line + "\n").encode("utf-8"))

    async with OllamaClient(transport=httpx.MockTransport(handler), single_flight=True) as client:
        t_hold = asyncio.create_task(
            client.chat("qwen3:1.7b", [{"role": "user", "content": "hold"}], priority=RequestPriority.NORMAL)
        )
        await asyncio.sleep(0.01)
        t_bg = asyncio.create_task(
            client.chat("qwen3:1.7b", [{"role": "user", "content": "bg"}], priority=RequestPriority.BACKGROUND)
        )
        t_hi = asyncio.create_task(
            client.chat("qwen3:1.7b", [{"role": "user", "content": "high"}], priority=RequestPriority.HIGH)
        )
        await asyncio.gather(t_hold, t_bg, t_hi)

    assert order == ["hold", "high", "bg"]
