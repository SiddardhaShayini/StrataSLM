"""Single-process FastAPI gateway for StrataSLM (/v1/chat/completions, /v1/runs, /v1/memory)."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Sequence

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from strata.agents.journal import DurableStepJournal
from strata.agents.runtime import DeterministicAgentRuntime, RegisteredTool
from strata.config import StrataConfig
from strata.governor.resource_governor import ResourceGovernor
from strata.inference.client import OllamaClient, RequestPriority
from strata.inference.engine import InferenceEngine, ResourcePressureError
from strata.memory.consolidator import SleepTimeConsolidator
from strata.memory.embedder import create_embedder
from strata.memory.store import CognitiveMemoryStore
from strata.plugins.loader import PluginRegistry
from strata.storage.db import StrataDB


class ChatMessageIn(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessageIn]
    stream: bool = False
    temperature: float = 0.2
    max_tokens: int = 256
    think: bool | None = None
    use_cache: bool = True
    use_semantic_cache: bool = True
    session_id: str | None = None
    recall_memory: bool = True


class AgentRunRequest(BaseModel):
    goal: str
    resume_run_id: str | None = None
    max_steps: int = 6
    rollback_on_failure: bool = True


class EpisodeIn(BaseModel):
    session_id: str
    role: str = "user"
    content: str
    importance: float = 0.6


class RecallRequest(BaseModel):
    query: str
    top_k: int = 5
    as_of: float | None = None


class ConsolidateRequest(BaseModel):
    batch_size: int = 10


def _split_messages(
    messages: Sequence[ChatMessageIn],
) -> tuple[str | None, list[dict[str, str]], str]:
    system_parts: list[str] = []
    history: list[dict[str, str]] = []
    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        else:
            history.append({"role": msg.role, "content": msg.content})

    if not history:
        raise HTTPException(status_code=400, detail="At least one non-system message is required.")

    last_user_idx = len(history) - 1
    for i in range(len(history) - 1, -1, -1):
        if history[i]["role"] == "user":
            last_user_idx = i
            break
    query = history[last_user_idx]["content"]
    prior_history = history[:last_user_idx] + history[last_user_idx + 1 :]
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return system_prompt, prior_history, query


def create_app(
    config: StrataConfig | None = None,
    *,
    tools: Sequence[RegisteredTool] = (),
    plugins_dir: Path | str | None = None,
    workspace_root: Path | None = None,
    client: OllamaClient | None = None,
    enable_background_sleep: bool = False,
    sleep_interval_sec: float = 30.0,
) -> FastAPI:
    """Create the lightweight single-process FastAPI gateway."""
    cfg = config or StrataConfig.from_env()
    registry = PluginRegistry(workspace_root=workspace_root)
    if plugins_dir is not None:
        registry.load_directory(plugins_dir)
    for t in tools:
        registry.register_tool(t)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        db = StrataDB(cfg.db_path)
        embedder = create_embedder(
            prefer_onnx=True,
            model_dir=cfg.data_dir / "models" / "all-MiniLM-L6-v2",
            num_threads=cfg.num_thread,
        )
        governor = ResourceGovernor(cfg)
        ollama_client = client or OllamaClient(cfg.ollama_host)
        owns_client = client is None
        engine = InferenceEngine(
            cfg, db=db, client=ollama_client, governor=governor, embedder=embedder
        )
        memory = CognitiveMemoryStore(db, embedder=embedder)
        consolidator = SleepTimeConsolidator(memory, governor)
        journal = DurableStepJournal(db)

        app.state.config = cfg
        app.state.db = db
        app.state.governor = governor
        app.state.engine = engine
        app.state.memory = memory
        app.state.consolidator = consolidator
        app.state.journal = journal
        app.state.registry = registry

        sleep_task: asyncio.Task[None] | None = None
        if enable_background_sleep:
            async def _sleep_loop() -> None:
                while True:
                    await asyncio.sleep(sleep_interval_sec)
                    try:
                        await consolidator.run_once(batch_size=5)
                    except Exception:
                        pass

            sleep_task = asyncio.create_task(_sleep_loop())

        try:
            yield
        finally:
            if sleep_task is not None:
                sleep_task.cancel()
            if owns_client:
                await ollama_client.aclose()
            db.close()

    app = FastAPI(
        title="StrataSLM Runtime Gateway",
        version="0.1.0",
        description="Lightweight OpenAI-compatible local AI runtime & cognitive memory API",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        assess = app.state.governor.evaluate(sample_interval=None)
        ollama_health = await app.state.engine.client.health()
        return {
            "status": "ok" if assess.allow_new_requests else "degraded",
            "governor": assess.to_dict(),
            "ollama": ollama_health.to_dict(),
            "has_vec": app.state.db.has_vec,
            "registered_tools": [t.spec.name for t in app.state.registry.list_tools()],
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        ollama_health = await app.state.engine.client.health()
        names = [m.name for m in ollama_health.installed_models] or [
            cfg.primary_model,
            cfg.router_model,
        ]
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {"id": n, "object": "model", "created": now, "owned_by": "strata-local"}
                for n in names
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> Any:
        system_prompt, history, query = _split_messages(req.messages)
        memory: CognitiveMemoryStore = app.state.memory
        engine: InferenceEngine = app.state.engine

        memory_facts: list[str] = []
        if req.recall_memory:
            memory_facts = [f.fact_text for f in memory.recall_facts(query, top_k=4)]
        if req.session_id and not history:
            history = memory.get_working_turns(req.session_id)

        try:
            res = await engine.complete(
                query,
                system_prompt=system_prompt,
                memory_facts=memory_facts,
                history=history,
                use_cache=req.use_cache,
                use_semantic_cache=req.use_semantic_cache,
                force_think=req.think,
                force_model=req.model,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                priority=RequestPriority.HIGH,
            )
        except ResourcePressureError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        if req.session_id:
            memory.push_working_turn(req.session_id, "user", query)
            memory.push_working_turn(req.session_id, "assistant", res.content)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if not req.stream:
            prompt_toks = res.metrics.prompt_eval_count if res.metrics else res.packed.estimated_tokens
            comp_toks = res.metrics.eval_count if res.metrics else max(1, len(res.content) // 4)
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": res.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": res.content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_toks,
                    "completion_tokens": comp_toks,
                    "total_tokens": prompt_toks + comp_toks,
                },
                "strata": {
                    "cached": res.cached,
                    "cache_tier": res.cache_tier,
                    "cache_latency_ms": res.cache_latency_ms,
                    "cache_similarity": res.cache_similarity,
                    "coalesced": res.coalesced,
                    "think": res.think,
                    "governor_state": res.governor_state,
                    "prefix_hash": res.packed.prefix_hash,
                },
            }

        async def _sse_stream() -> AsyncIterator[str]:
            first = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": res.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": res.content},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(first)}\n\n"
            final = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": res.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_sse_stream(), media_type="text/event-stream")

    @app.post("/v1/runs")
    async def create_agent_run(req: AgentRunRequest) -> dict[str, Any]:
        memory: CognitiveMemoryStore = app.state.memory
        facts = [f.fact_text for f in memory.recall_facts(req.goal, top_k=4)]
        rt = DeterministicAgentRuntime(
            app.state.engine,
            app.state.registry.list_tools(),
            workspace_root=workspace_root,
            memory=memory,
            max_steps=req.max_steps,
        )
        res = await rt.run(
            req.goal,
            resume_run_id=req.resume_run_id,
            memory_facts=facts,
            rollback_on_failure=req.rollback_on_failure,
        )
        return res.to_dict()

    @app.get("/v1/runs/{run_id}")
    async def get_agent_run(run_id: str) -> dict[str, Any]:
        journal: DurableStepJournal = app.state.journal
        run = journal.load_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
        steps = [s.to_dict() for s in journal.list_steps(run_id)]
        return {
            "run_id": run.run_id,
            "goal": run.goal,
            "status": run.status.value,
            "current_state": run.current_state.value,
            "step_index": run.step_index,
            "steps": steps,
        }

    @app.post("/v1/memory/episodes")
    async def add_episode(ep: EpisodeIn) -> dict[str, Any]:
        eid = app.state.memory.add_episode(
            ep.session_id, ep.role, ep.content, importance=ep.importance
        )
        return {"episode_id": eid}

    @app.post("/v1/memory/recall")
    async def recall_memory(req: RecallRequest) -> dict[str, Any]:
        facts = app.state.memory.recall_facts(
            req.query, top_k=req.top_k, as_of=req.as_of
        )
        return {"facts": [f.to_dict() for f in facts]}

    @app.post("/v1/memory/consolidate")
    async def consolidate_memory(req: ConsolidateRequest) -> dict[str, Any]:
        rep = await app.state.consolidator.run_once(batch_size=req.batch_size)
        return rep.to_dict()

    return app
