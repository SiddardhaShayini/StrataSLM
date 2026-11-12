"""Tests for `run_doctor`, `run_thread_sweep`, and CLI parser."""

import json
from pathlib import Path

import httpx
import pytest

from strata.cli import build_parser
from strata.config import StrataConfig
from strata.inference.client import OllamaClient
from strata.perf.bench import run_doctor, run_thread_sweep


def _mock_ollama_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/version":
        return httpx.Response(200, json={"version": "0.40.16"})
    if request.url.path == "/api/tags":
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "qwen3:1.7b", "size": 1_400_000_000, "details": {}},
                    {"name": "qwen3:0.6b", "size": 523_000_000, "details": {}},
                ]
            },
        )
    if request.url.path == "/api/ps":
        return httpx.Response(200, json={"models": []})
    if request.url.path == "/api/generate":
        return httpx.Response(200, json={"done": True})
    if request.url.path == "/api/chat":
        body = json.loads(request.content.decode("utf-8"))
        threads = (body.get("options") or {}).get("num_thread", 4)
        # Simulate fastest eval at 4 threads
        eval_ns = 100_000_000 if threads == 4 else 200_000_000
        line = json.dumps(
            {
                "model": body["model"],
                "message": {"role": "assistant", "content": "ok"},
                "done": True,
                "done_reason": "stop",
                "load_duration": 5_000_000,
                "prompt_eval_count": 30,
                "prompt_eval_duration": 150_000_000,
                "eval_count": 20,
                "eval_duration": eval_ns,
            }
        )
        return httpx.Response(200, content=(line + "\n").encode("utf-8"))
    return httpx.Response(404)


@pytest.mark.asyncio
async def test_doctor_and_thread_sweep(tmp_path: Path) -> None:
    cfg = StrataConfig()
    transport = httpx.MockTransport(_mock_ollama_handler)
    async with OllamaClient(transport=transport) as client:
        doc = await run_doctor(cfg, client=client)
        assert doc.ollama.reachable is True
        assert doc.passed is True

        sweep = await run_thread_sweep(
            cfg,
            model="qwen3:1.7b",
            threads=[2, 4],
            trials=2,
            save=True,
            results_dir=tmp_path,
            client=client,
        )
        assert len(sweep.results) == 2
        assert sweep.recommended_num_thread == 4
        assert sweep.saved_to is not None
        assert Path(sweep.saved_to).exists()


def test_cli_parser() -> None:
    parser = build_parser()
    args = parser.parse_args(["doctor", "--json"])
    assert args.command == "doctor"
    assert args.json is True

    args2 = parser.parse_args(["bench", "threads", "--threads", "2,4,6", "--think", "false"])
    assert args2.command == "bench"
    assert args2.bench_command == "threads"
    assert args2.threads == "2,4,6"

    args3 = parser.parse_args(["bench", "cold-start", "--runs", "3", "--json"])
    assert args3.command == "bench"
    assert args3.bench_command == "cold-start"
    assert args3.runs == 3
    assert args3.json is True

    args4 = parser.parse_args(["bench", "inference", "--model", "qwen3:1.7b", "--save"])
    assert args4.command == "bench"
    assert args4.bench_command == "inference"
    assert args4.model == "qwen3:1.7b"
    assert args4.save is True

    args5 = parser.parse_args(["profile", "--save", "--json"])
    assert args5.command == "profile"
    assert args5.save is True
    assert args5.json is True

    args6 = parser.parse_args(["models", "fetch-minilm", "--force", "--json"])
    assert args6.command == "models"
    assert args6.models_command == "fetch-minilm"
    assert args6.force is True
    assert args6.json is True

    args7 = parser.parse_args(["bench", "embedders", "--require-onnx", "--save"])
    assert args7.command == "bench"
    assert args7.bench_command == "embedders"
    assert args7.require_onnx is True
    assert args7.save is True


def test_eval_embedder_ab_missing_weights_graceful(tmp_path: Path) -> None:
    from strata.perf.evals import eval_embedder_ab

    rep = eval_embedder_ab(
        model_dir=tmp_path / "absent_minilm",
        save=True,
        results_dir=tmp_path,
        require_onnx=False,
    )
    assert rep["status"] == "SKIP"
    assert rep["passed"] is True
    assert rep["hashing"]["available"] is True
    assert rep["onnx_minilm"]["available"] is False
    assert (tmp_path / "embedder_ab_latest.json").exists()

    strict = eval_embedder_ab(
        model_dir=tmp_path / "absent_minilm",
        require_onnx=True,
    )
    assert strict["status"] == "FAIL"
    assert strict["passed"] is False


@pytest.mark.asyncio
async def test_doctor_unreachable_ollama_fails() -> None:
    cfg = StrataConfig(ollama_host="http://127.0.0.1:1")

    def fail_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with OllamaClient(transport=httpx.MockTransport(fail_handler)) as client:
        doc = await run_doctor(cfg, client=client)
        assert doc.ollama.reachable is False
        assert doc.passed is False
        payload = doc.to_dict()
        assert payload["passed"] is False
        assert any(c["name"] == "ollama_server" and c["status"] == "FAIL" for c in payload["checks"])
