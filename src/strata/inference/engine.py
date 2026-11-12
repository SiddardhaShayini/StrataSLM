"""Coordinated StrataSLM inference engine: governor -> prune -> pack -> route -> cache -> Ollama."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Sequence

from strata.config import StrataConfig
from strata.governor.resource_governor import (
    GovernorAssessment,
    GovernorState,
    ResourceGovernor,
)
from strata.inference.cache import L1ExactCache, L2SemanticCache, build_l1_key
from strata.inference.client import ChatMetrics, OllamaClient, RequestPriority
from strata.inference.prompt import (
    CanonicalPromptBuilder,
    ContextBudget,
    PackedPrompt,
)
from strata.inference.router import AdaptiveRouter, RouteDecision
from strata.inference.tool_pruner import PrunedToolsResult, ToolPruner, ToolSpec
from strata.memory.embedder import TextEmbedder
from strata.storage.db import StrataDB

DEFAULT_SYSTEM_PROMPT = (
    "You are StrataSLM, a concise, accurate local AI assistant running on an 8 GB CPU laptop. "
    "Follow instructions directly, prefer verified facts, and keep outputs structured and brief."
)


class ResourcePressureError(RuntimeError):
    """Raised when ResourceGovernor enters RED state and sheds uncached LLM work."""

    def __init__(self, assessment: GovernorAssessment) -> None:
        self.assessment = assessment
        super().__init__(
            f"Inference shed under RED resource pressure: {'; '.join(assessment.reasons)}"
        )


@dataclass(slots=True)
class InferenceResponse:
    content: str
    thinking: str
    model: str
    think: bool
    num_ctx: int
    cached: bool
    cache_tier: str | None
    cache_latency_ms: float | None
    route: RouteDecision
    packed: PackedPrompt
    pruned_tools: PrunedToolsResult | None
    metrics: ChatMetrics | None
    governor_state: str
    cache_similarity: float | None = None
    coalesced: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "thinking": self.thinking,
            "model": self.model,
            "think": self.think,
            "num_ctx": self.num_ctx,
            "cached": self.cached,
            "cache_tier": self.cache_tier,
            "cache_latency_ms": self.cache_latency_ms,
            "cache_similarity": self.cache_similarity,
            "coalesced": self.coalesced,
            "route": self.route.to_dict(),
            "packed": self.packed.to_dict(),
            "pruned_tools": self.pruned_tools.to_dict() if self.pruned_tools else None,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "governor_state": self.governor_state,
        }


def _compute_context_hash(
    memory_facts: Sequence[str], history: Sequence[dict[str, str]]
) -> str:
    if not memory_facts and not history:
        return ""
    blob = json.dumps(
        {"memory": list(memory_facts), "history": list(history)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class InferenceEngine:
    """Single-process inference coordinator optimizing KV reuse, caching, and 8 GB safety."""

    def __init__(
        self,
        config: StrataConfig,
        *,
        db: StrataDB | None = None,
        client: OllamaClient | None = None,
        governor: ResourceGovernor | None = None,
        embedder: TextEmbedder | None = None,
        l2_similarity_threshold: float = 0.88,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.config = config
        self._owns_db = db is None
        self.db = db or StrataDB(config.db_path)
        self._owns_client = client is None
        self.client = client or OllamaClient(config.ollama_host)
        self.governor = governor or ResourceGovernor(config)
        self.router = AdaptiveRouter(config)
        self.cache = L1ExactCache(self.db)
        self.l2_cache = L2SemanticCache(
            self.db,
            embedder=embedder,
            similarity_threshold=l2_similarity_threshold,
        )
        self._inflight: dict[str, asyncio.Future[InferenceResponse]] = {}
        self.default_system_prompt = system_prompt

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()
        if self._owns_db:
            self.db.close()

    async def __aenter__(self) -> InferenceEngine:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def complete(
        self,
        query: str,
        *,
        system_prompt: str | None = None,
        tools: Sequence[ToolSpec] = (),
        memory_facts: Sequence[str] = (),
        history: Sequence[dict[str, str]] = (),
        top_k_tools: int = 4,
        use_cache: bool = True,
        use_semantic_cache: bool = True,
        force_think: bool | None = None,
        force_model: str | None = None,
        force_num_ctx: int | None = None,
        prefer_fast_model: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.2,
        priority: RequestPriority | int = RequestPriority.NORMAL,
    ) -> InferenceResponse:
        assessment = self.governor.evaluate(sample_interval=None)
        if assessment.should_trim_caches:
            self.cache.trim_memory(0)

        pruned: PrunedToolsResult | None = None
        selected_tools: Sequence[ToolSpec] = ()
        schema_hash = "no_tools"
        if tools:
            pruned = ToolPruner(tools).select(query, top_k=top_k_tools)
            selected_tools = pruned.selected
            schema_hash = pruned.schema_hash

        decision = self.router.route(
            query,
            tool_count=len(selected_tools),
            history_turns=len(history),
            governor_state=assessment.state,
            prefer_fast_model=prefer_fast_model,
        )
        model = force_model or decision.model
        think = decision.think if force_think is None else force_think
        num_ctx = force_num_ctx or decision.num_ctx

        builder = CanonicalPromptBuilder(
            system_prompt or self.default_system_prompt,
            budget=ContextBudget(num_ctx=num_ctx, reserved_output_tokens=max_tokens),
        )
        packed = builder.pack(
            query,
            tools=selected_tools,
            memory_facts=memory_facts,
            history=history,
        )
        context_hash = _compute_context_hash(memory_facts, packed.messages[1:-1])

        if use_cache:
            hit = self.cache.get(
                model=model,
                think=think,
                prefix_hash=packed.prefix_hash,
                schema_hash=schema_hash,
                query=query,
                context_hash=context_hash,
            )
            if not hit.hit and use_semantic_cache:
                hit = self.l2_cache.get(
                    model=model,
                    think=think,
                    prefix_hash=packed.prefix_hash,
                    schema_hash=schema_hash,
                    query=query,
                    context_hash=context_hash,
                )
            if hit.hit and hit.payload is not None:
                return InferenceResponse(
                    content=str(hit.payload.get("content") or ""),
                    thinking=str(hit.payload.get("thinking") or ""),
                    model=model,
                    think=think,
                    num_ctx=num_ctx,
                    cached=True,
                    cache_tier=hit.tier,
                    cache_latency_ms=hit.latency_ms,
                    cache_similarity=hit.similarity,
                    route=decision,
                    packed=packed,
                    pruned_tools=pruned,
                    metrics=None,
                    governor_state=assessment.state.value,
                )

        if assessment.state == GovernorState.RED or not assessment.allow_new_requests:
            raise ResourcePressureError(assessment)

        inflight_key, _ = build_l1_key(
            model=model,
            think=think,
            prefix_hash=packed.prefix_hash,
            schema_hash=schema_hash,
            query=query,
            context_hash=context_hash,
        )
        existing_fut = self._inflight.get(inflight_key)
        if existing_fut is not None and not existing_fut.done():
            shared = await asyncio.shield(existing_fut)
            return replace(shared, coalesced=True)

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[InferenceResponse] = loop.create_future()
        self._inflight[inflight_key] = fut
        try:
            result = await self.client.chat(
                model,
                packed.messages,
                think=think,
                num_ctx=num_ctx,
                num_thread=self.config.num_thread,
                temperature=temperature,
                max_tokens=max_tokens,
                keep_alive=self.config.keep_alive,
                stream=True,
                priority=priority,
            )

            if use_cache:
                payload = {"content": result.content, "thinking": result.thinking}
                self.cache.put(
                    model=model,
                    think=think,
                    prefix_hash=packed.prefix_hash,
                    schema_hash=schema_hash,
                    query=query,
                    context_hash=context_hash,
                    payload=payload,
                )
                if use_semantic_cache:
                    self.l2_cache.put(
                        model=model,
                        think=think,
                        prefix_hash=packed.prefix_hash,
                        schema_hash=schema_hash,
                        query=query,
                        context_hash=context_hash,
                        payload=payload,
                    )

            response = InferenceResponse(
                content=result.content,
                thinking=result.thinking,
                model=result.model,
                think=think,
                num_ctx=num_ctx,
                cached=False,
                cache_tier=None,
                cache_latency_ms=None,
                route=decision,
                packed=packed,
                pruned_tools=pruned,
                metrics=result.metrics,
                governor_state=assessment.state.value,
            )
            if not fut.done():
                fut.set_result(response)
            return response
        except BaseException as exc:
            if not fut.done():
                fut.set_exception(exc)
            raise
        finally:
            self._inflight.pop(inflight_key, None)
