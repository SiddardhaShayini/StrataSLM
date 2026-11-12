"""Tests for FastAPI gateway (`strata.server`), SDK (`@tool`, `StrataClient`), and `PluginRegistry`."""

import json
import sys
from pathlib import Path

import httpx
import pytest

from strata.agents.policy import Capability
from strata.config import StrataConfig
from strata.inference.client import OllamaClient
from strata.plugins.loader import PluginRegistry, adapt_mcp_tool
from strata.sdk import StrataClient, tool
from strata.server.app import create_app


@tool(tags=("math",), capability=Capability.COMPUTE)
def add_numbers(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b


def test_tool_decorator_and_plugin_manifest(tmp_path: Path) -> None:
    assert add_numbers(2, 3) == 5
    reg = add_numbers.strata_tool
    assert reg.spec.name == "add_numbers"
    assert reg.spec.parameters["required"] == ["a", "b"]
    assert reg.spec.parameters["properties"]["a"]["type"] == "integer"

    # Write a manifest plugin using sandboxed Python subprocess
    manifest_file = tmp_path / "echo_plugin.json"
    manifest_file.write_text(
        json.dumps(
            {
                "name": "echo_plugin",
                "version": "0.1.0",
                "tools": [
                    {
                        "name": "say_hello",
                        "description": "Echo greeting inside sandboxed subprocess",
                        "capability": "read",
                        "tags": ["demo"],
                        "parameters": {
                            "type": "object",
                            "properties": {"who": {"type": "string"}},
                            "required": ["who"],
                        },
                        "command": [sys.executable, "-c", "print('hello ' + {who!r})"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    registry = PluginRegistry(workspace_root=tmp_path)
    count = registry.load_directory(tmp_path)
    assert count == 1
    tools = registry.list_tools()
    assert [t.spec.name for t in tools] == ["say_hello"]
    out = tools[0].handler({"who": "strata"})
    assert out["ok"] is True
    assert "hello strata" in out["stdout"]

    # Null-byte injection in templated subprocess arguments must be rejected
    with pytest.raises(ValueError, match="(?i)null byte"):
        tools[0].handler({"who": "strata\x00evil"})

    mcp_tool = adapt_mcp_tool(
        {"name": "fetch_ticket", "description": "Read external ticket", "inputSchema": {"type": "object", "properties": {}}},
        lambda name, args: f"{name}:ok",
    )
    assert mcp_tool.returns_untrusted is True


@pytest.mark.asyncio
async def test_fastapi_gateway_and_remote_sdk(tmp_path: Path) -> None:
    seq = iter(
        [
            '{"decision":"call_tool","tool_name":"add_numbers","arguments":{"a":4,"b":7}}',
            '{"decision":"final_answer","final_answer":"Sum is 11"}',
            "Paris is the capital of France.",
            "Streaming reply.",
        ]
    )

    def ollama_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.16"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:1.7b", "size": 1_400_000_000}]})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/chat":
            line = json.dumps(
                {
                    "model": "qwen3:1.7b",
                    "message": {"role": "assistant", "content": next(seq)},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 12,
                    "prompt_eval_duration": 50_000_000,
                    "eval_count": 6,
                    "eval_duration": 50_000_000,
                }
            )
            return httpx.Response(200, content=(line + "\n").encode("utf-8"))
        return httpx.Response(404)

    cfg = StrataConfig(data_dir=tmp_path)
    async with OllamaClient(transport=httpx.MockTransport(ollama_handler)) as mock_ollama:
        app = create_app(
            cfg,
            tools=[add_numbers.strata_tool],
            workspace_root=tmp_path,
            client=mock_ollama,
            enable_background_sleep=False,
        )
        # Enter FastAPI lifespan cleanly and bind StrataClient.remote to ASGITransport
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with StrataClient.remote("http://testserver", transport=transport) as sdk:
                health = await sdk.health()
                assert health["status"] == "ok"

                # Agent run via /v1/runs + /v1/runs/{run_id}
                run_res = await sdk.run_agent("Add 4 and 7")
                assert run_res["status"] == "completed"
                assert run_res["final_answer"] == "Sum is 11"

                # Memory episodes -> consolidate -> recall via HTTP endpoints
                await sdk.remember("s1", "user", "My name is Kiran. I live in Kochi.")
                cons = await sdk.consolidate()
                assert cons["facts_added"] >= 2
                recalled = await sdk.recall("Where does Kiran live?")
                assert any(f["object_value"] == "Kochi" for f in recalled)

                # Chat completions (miss + L1 hit)
                c1 = await sdk.chat("What is the capital of France?", recall_memory=False)
                assert c1["choices"][0]["message"]["content"] == "Paris is the capital of France."
                assert c1["strata"]["cached"] is False

                c2 = await sdk.chat("What is the capital of France?", recall_memory=False)
                assert c2["strata"]["cached"] is True
                assert c2["strata"]["cache_tier"] == "L1_MEM"

            # Also verify SSE streaming on /v1/chat/completions
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as raw_http:
                sse_resp = await raw_http.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "Stream test"}],
                        "stream": True,
                        "recall_memory": False,
                    },
                )
                assert sse_resp.status_code == 200
                assert "text/event-stream" in sse_resp.headers["content-type"]
                assert "Streaming reply." in sse_resp.text
                assert "data: [DONE]" in sse_resp.text


@pytest.mark.asyncio
async def test_strata_client_local_memory_and_consolidation(tmp_path: Path) -> None:
    cfg = StrataConfig(data_dir=tmp_path)
    async with StrataClient.local(cfg, prefer_onnx=False) as sdk:
        await sdk.remember("s1", "user", "I prefer concise SQL queries.")
        rep = await sdk.consolidate()
        assert rep["facts_added"] == 1
        recalled = await sdk.recall("What does the user prefer?")
        assert any("concise SQL queries" in f["object_value"] for f in recalled)
