"""Async Ollama HTTP client with streaming TTFT telemetry and priority single-flight guard."""

from __future__ import annotations

import asyncio
import heapq
import json
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, AsyncIterator

import httpx


class RequestPriority(IntEnum):
    """Scheduling lanes for single-flight Ollama requests (lower value runs sooner)."""

    HIGH = 0
    NORMAL = 1
    BACKGROUND = 2


class _PriorityGate:
    """Single-slot async mutex draining queued waiters by (priority, sequence)."""

    def __init__(self) -> None:
        self._busy = False
        self._seq = 0
        self._waiters: list[tuple[int, int, asyncio.Future[None]]] = []

    @asynccontextmanager
    async def acquire(self, priority: int = RequestPriority.NORMAL) -> AsyncIterator[None]:
        if not self._busy and not self._waiters:
            self._busy = True
        else:
            loop = asyncio.get_running_loop()
            fut: asyncio.Future[None] = loop.create_future()
            self._seq += 1
            heapq.heappush(self._waiters, (int(priority), self._seq, fut))
            try:
                await fut
            except BaseException:
                if not fut.done():
                    fut.cancel()
                elif not fut.cancelled():
                    self._release()
                raise
        try:
            yield
        finally:
            self._release()

    def _release(self) -> None:
        while self._waiters:
            _, _, fut = heapq.heappop(self._waiters)
            if not fut.done():
                fut.set_result(None)
                return
        self._busy = False


@dataclass(slots=True)
class InstalledModel:
    name: str
    size_bytes: int
    parameter_size: str | None = None
    quantization: str | None = None

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / (1024.0**3), 2)


@dataclass(slots=True)
class LoadedModel:
    name: str
    size_bytes: int
    size_vram_bytes: int
    context_length: int | None = None


@dataclass(slots=True)
class OllamaHealth:
    reachable: bool
    host: str
    version: str | None = None
    installed_models: list[InstalledModel] = field(default_factory=list)
    loaded_models: list[LoadedModel] = field(default_factory=list)
    error: str | None = None

    def has_model(self, name: str) -> bool:
        target = name.lower()
        return any(
            m.name.lower() == target or m.name.lower().startswith(f"{target}:")
            for m in self.installed_models
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reachable": self.reachable,
            "host": self.host,
            "version": self.version,
            "installed_models": [asdict(m) for m in self.installed_models],
            "loaded_models": [asdict(m) for m in self.loaded_models],
            "error": self.error,
        }


@dataclass(slots=True)
class ChatMetrics:
    """Measured client-side and server-reported Ollama inference metrics."""

    wall_ms: float
    ttft_ms: float
    first_content_ms: float | None
    load_duration_ms: float
    prompt_eval_count: int
    prompt_eval_duration_ms: float
    eval_count: int
    eval_duration_ms: float
    prompt_tok_per_sec: float
    eval_tok_per_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChatResult:
    model: str
    content: str
    thinking: str
    done_reason: str | None
    metrics: ChatMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "content": self.content,
            "thinking": self.thinking,
            "done_reason": self.done_reason,
            "metrics": self.metrics.to_dict(),
        }


def _ns_to_ms(ns: int | float | None) -> float:
    if not ns:
        return 0.0
    return round(float(ns) / 1_000_000.0, 2)


def _tok_per_sec(tokens: int, duration_ns: int | float | None) -> float:
    if tokens <= 0 or not duration_ns or duration_ns <= 0:
        return 0.0
    return round(tokens / (float(duration_ns) / 1_000_000_000.0), 2)


class OllamaClient:
    """Lightweight async client for local Ollama REST API with priority single-flight guard."""

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        *,
        timeout: float = 120.0,
        single_flight: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._gate = _PriorityGate() if single_flight else None
        self._client = httpx.AsyncClient(
            base_url=self._host,
            timeout=httpx.Timeout(timeout, connect=5.0),
            transport=transport,
        )

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> OllamaHealth:
        """Query Ollama version, installed tags, and currently loaded runners."""
        try:
            ver_resp = await self._client.get("/api/version")
            ver_resp.raise_for_status()
            version = ver_resp.json().get("version")

            tags_resp = await self._client.get("/api/tags")
            tags_resp.raise_for_status()
            installed = [
                InstalledModel(
                    name=m.get("name", ""),
                    size_bytes=int(m.get("size", 0)),
                    parameter_size=(m.get("details") or {}).get("parameter_size"),
                    quantization=(m.get("details") or {}).get("quantization_level"),
                )
                for m in tags_resp.json().get("models", [])
            ]

            ps_resp = await self._client.get("/api/ps")
            ps_resp.raise_for_status()
            loaded = [
                LoadedModel(
                    name=m.get("name", ""),
                    size_bytes=int(m.get("size", 0)),
                    size_vram_bytes=int(m.get("size_vram", 0)),
                    context_length=m.get("context_length"),
                )
                for m in ps_resp.json().get("models", [])
            ]

            return OllamaHealth(
                reachable=True,
                host=self._host,
                version=version,
                installed_models=installed,
                loaded_models=loaded,
            )
        except Exception as exc:
            return OllamaHealth(
                reachable=False,
                host=self._host,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def unload_model(
        self,
        model: str,
        *,
        priority: RequestPriority | int = RequestPriority.BACKGROUND,
    ) -> None:
        """Instruct Ollama to immediately unload a model runner (`keep_alive=0`)."""
        payload = {"model": model, "keep_alive": 0}
        if self._gate is None:
            resp = await self._client.post("/api/generate", json=payload)
            resp.raise_for_status()
            return
        async with self._gate.acquire(priority):
            resp = await self._client.post("/api/generate", json=payload)
            resp.raise_for_status()

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        think: bool | None = None,
        num_ctx: int = 4096,
        num_thread: int | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        keep_alive: str | int = "5m",
        stream: bool = True,
        priority: RequestPriority | int = RequestPriority.NORMAL,
    ) -> ChatResult:
        """Send a `/api/chat` request and collect text + fine-grained latency metrics."""
        options: dict[str, Any] = {
            "num_ctx": num_ctx,
            "temperature": temperature,
        }
        if num_thread is not None:
            options["num_thread"] = num_thread
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "keep_alive": keep_alive,
            "options": options,
        }
        if think is not None:
            payload["think"] = think

        if self._gate is None:
            return await self._send_chat(model, payload, stream=stream)
        async with self._gate.acquire(priority):
            return await self._send_chat(model, payload, stream=stream)

    async def _send_chat(
        self, model: str, payload: dict[str, Any], *, stream: bool
    ) -> ChatResult:
        start = time.perf_counter()
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        first_token_ms: float | None = None
        first_content_ms: float | None = None
        final_chunk: dict[str, Any] = {}

        if stream:
            async with self._client.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    msg = chunk.get("message") or {}
                    tok_content = msg.get("content") or ""
                    tok_thinking = msg.get("thinking") or ""
                    now_ms = (time.perf_counter() - start) * 1000.0
                    if tok_thinking:
                        if first_token_ms is None:
                            first_token_ms = now_ms
                        thinking_parts.append(tok_thinking)
                    if tok_content:
                        if first_token_ms is None:
                            first_token_ms = now_ms
                        if first_content_ms is None:
                            first_content_ms = now_ms
                        content_parts.append(tok_content)
                    if chunk.get("done"):
                        final_chunk = chunk
        else:
            resp = await self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            final_chunk = resp.json()
            msg = final_chunk.get("message") or {}
            if msg.get("thinking"):
                thinking_parts.append(msg["thinking"])
            if msg.get("content"):
                content_parts.append(msg["content"])
            wall_now = (time.perf_counter() - start) * 1000.0
            first_token_ms = wall_now
            first_content_ms = wall_now if content_parts else None

        wall_ms = (time.perf_counter() - start) * 1000.0
        if first_token_ms is None:
            first_token_ms = wall_ms

        prompt_eval_count = int(final_chunk.get("prompt_eval_count") or 0)
        prompt_eval_ns = int(final_chunk.get("prompt_eval_duration") or 0)
        eval_count = int(final_chunk.get("eval_count") or 0)
        eval_ns = int(final_chunk.get("eval_duration") or 0)
        load_ns = int(final_chunk.get("load_duration") or 0)

        metrics = ChatMetrics(
            wall_ms=round(wall_ms, 2),
            ttft_ms=round(first_token_ms, 2),
            first_content_ms=(
                round(first_content_ms, 2) if first_content_ms is not None else None
            ),
            load_duration_ms=_ns_to_ms(load_ns),
            prompt_eval_count=prompt_eval_count,
            prompt_eval_duration_ms=_ns_to_ms(prompt_eval_ns),
            eval_count=eval_count,
            eval_duration_ms=_ns_to_ms(eval_ns),
            prompt_tok_per_sec=_tok_per_sec(prompt_eval_count, prompt_eval_ns),
            eval_tok_per_sec=_tok_per_sec(eval_count, eval_ns),
        )

        return ChatResult(
            model=final_chunk.get("model") or model,
            content="".join(content_parts),
            thinking="".join(thinking_parts),
            done_reason=final_chunk.get("done_reason"),
            metrics=metrics,
        )
