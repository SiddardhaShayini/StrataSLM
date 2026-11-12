"""Python SDK providing `@tool` schema extraction and unified local/remote `StrataClient`."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Sequence, get_type_hints

import httpx

from strata.agents.policy import Capability
from strata.agents.runtime import AgentRunResult, DeterministicAgentRuntime, RegisteredTool
from strata.config import StrataConfig
from strata.governor.resource_governor import ResourceGovernor
from strata.inference.engine import InferenceEngine, InferenceResponse
from strata.inference.tool_pruner import ToolSpec
from strata.memory.consolidator import ConsolidationReport, SleepTimeConsolidator
from strata.memory.embedder import create_embedder
from strata.memory.store import CognitiveMemoryStore
from strata.storage.db import StrataDB

_TYPE_MAP: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type_for_annotation(ann: Any) -> str:
    origin = getattr(ann, "__origin__", None)
    if origin in _TYPE_MAP:
        return _TYPE_MAP[origin]
    return _TYPE_MAP.get(ann, "string")


def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    capability: Capability = Capability.READ,
    tags: Sequence[str] = (),
    always_include: bool = False,
    returns_untrusted: bool = False,
    compensator: Callable[[dict[str, Any], Any], Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator turning a typed Python function into a StrataSLM `RegisteredTool`."""

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or fn.__name__
        tool_desc = description or (inspect.getdoc(fn) or f"Invoke {tool_name}").splitlines()[0]
        sig = inspect.signature(fn)
        try:
            hints = get_type_hints(fn)
        except Exception:
            hints = {}

        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, param in sig.parameters.items():
            ann = hints.get(param_name, str)
            properties[param_name] = {"type": _json_type_for_annotation(ann)}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        spec = ToolSpec(
            name=tool_name,
            description=tool_desc,
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
            tags=tuple(tags),
            always_include=always_include,
            capability=capability.value,
        )

        async def _dispatch(args: dict[str, Any]) -> Any:
            res = fn(**args)
            if inspect.isawaitable(res):
                return await res
            return res

        reg = RegisteredTool(
            spec=spec,
            capability=capability,
            handler=_dispatch,
            returns_untrusted=returns_untrusted,
            compensator=compensator,
        )
        setattr(fn, "strata_tool", reg)
        setattr(fn, "tool_spec", spec)
        return fn

    return _decorator


def to_registered_tool(item: RegisteredTool | Callable[..., Any]) -> RegisteredTool:
    if isinstance(item, RegisteredTool):
        return item
    reg = getattr(item, "strata_tool", None)
    if isinstance(reg, RegisteredTool):
        return reg
    raise TypeError(f"Object {item!r} is neither a RegisteredTool nor decorated with @tool")


class StrataClient:
    """Unified embedded (zero-hop) or HTTP remote client for StrataSLM."""

    def __init__(
        self,
        *,
        mode: str,
        config: StrataConfig | None = None,
        db: StrataDB | None = None,
        engine: InferenceEngine | None = None,
        memory: CognitiveMemoryStore | None = None,
        consolidator: SleepTimeConsolidator | None = None,
        tools: Sequence[RegisteredTool | Callable[..., Any]] = (),
        workspace_root: Path | None = None,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.mode = mode
        self.config = config or StrataConfig.from_env()
        self._db = db
        self._engine = engine
        self._memory = memory
        self._consolidator = consolidator
        self._tools = [to_registered_tool(t) for t in tools]
        self._workspace_root = workspace_root
        self._base_url = (base_url or "").rstrip("/")
        self._http = http_client

    @classmethod
    def local(
        cls,
        config: StrataConfig | None = None,
        *,
        tools: Sequence[RegisteredTool | Callable[..., Any]] = (),
        workspace_root: Path | None = None,
        prefer_onnx: bool = True,
    ) -> StrataClient:
        """Create an embedded single-process client sharing one `strata.db`."""
        cfg = config or StrataConfig.from_env()
        db = StrataDB(cfg.db_path)
        embedder = create_embedder(
            prefer_onnx=prefer_onnx,
            model_dir=cfg.data_dir / "models" / "all-MiniLM-L6-v2",
            num_threads=cfg.num_thread,
        )
        gov = ResourceGovernor(cfg)
        engine = InferenceEngine(cfg, db=db, governor=gov, embedder=embedder)
        memory = CognitiveMemoryStore(db, embedder=embedder)
        consolidator = SleepTimeConsolidator(memory, gov)
        return cls(
            mode="local",
            config=cfg,
            db=db,
            engine=engine,
            memory=memory,
            consolidator=consolidator,
            tools=tools,
            workspace_root=workspace_root,
        )

    @classmethod
    def remote(
        cls,
        base_url: str = "http://127.0.0.1:8000",
        *,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> StrataClient:
        """Create an async HTTP client targeting a running `strata serve` server."""
        http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=5.0),
            transport=transport,
        )
        return cls(mode="remote", base_url=base_url, http_client=http)

    async def __aenter__(self) -> StrataClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._engine is not None:
            await self._engine.aclose()
        if self._db is not None:
            self._db.close()
        if self._http is not None:
            await self._http.aclose()

    async def chat(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        recall_memory: bool = True,
        use_cache: bool = True,
        think: bool | None = None,
        max_tokens: int = 256,
    ) -> dict[str, Any]:
        if self.mode == "local":
            assert self._engine is not None and self._memory is not None
            facts: list[str] = []
            history: list[dict[str, str]] = []
            if session_id:
                history = self._memory.get_working_turns(session_id)
                if recall_memory:
                    facts = [f.fact_text for f in self._memory.recall_facts(prompt, top_k=4)]
            res: InferenceResponse = await self._engine.complete(
                prompt,
                memory_facts=facts,
                history=history,
                use_cache=use_cache,
                force_think=think,
                max_tokens=max_tokens,
            )
            if session_id:
                self._memory.push_working_turn(session_id, "user", prompt)
                self._memory.push_working_turn(session_id, "assistant", res.content)
            return res.to_dict()

        assert self._http is not None
        resp = await self._http.post(
            "/v1/chat/completions",
            json={
                "model": self.config.primary_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "max_tokens": max_tokens,
                "think": think,
                "use_cache": use_cache,
                "session_id": session_id,
                "recall_memory": recall_memory,
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def run_agent(
        self,
        goal: str,
        *,
        tools: Sequence[RegisteredTool | Callable[..., Any]] | None = None,
        max_steps: int = 6,
    ) -> dict[str, Any]:
        if self.mode == "local":
            assert self._engine is not None and self._memory is not None
            active_tools = [to_registered_tool(t) for t in tools] if tools is not None else self._tools
            facts = [f.fact_text for f in self._memory.recall_facts(goal, top_k=4)]
            rt = DeterministicAgentRuntime(
                self._engine,
                active_tools,
                workspace_root=self._workspace_root,
                memory=self._memory,
                max_steps=max_steps,
            )
            res: AgentRunResult = await rt.run(goal, memory_facts=facts)
            return res.to_dict()

        assert self._http is not None
        resp = await self._http.post(
            "/v1/runs",
            json={"goal": goal, "max_steps": max_steps},
        )
        resp.raise_for_status()
        return resp.json()

    async def remember(
        self, session_id: str, role: str, content: str, *, importance: float = 0.6
    ) -> int:
        if self.mode == "local":
            assert self._memory is not None
            return self._memory.add_episode(session_id, role, content, importance=importance)
        assert self._http is not None
        resp = await self._http.post(
            "/v1/memory/episodes",
            json={
                "session_id": session_id,
                "role": role,
                "content": content,
                "importance": importance,
            },
        )
        resp.raise_for_status()
        return int(resp.json()["episode_id"])

    async def recall(
        self, query: str, *, top_k: int = 5, as_of: float | None = None
    ) -> list[dict[str, Any]]:
        if self.mode == "local":
            assert self._memory is not None
            return [f.to_dict() for f in self._memory.recall_facts(query, top_k=top_k, as_of=as_of)]
        assert self._http is not None
        resp = await self._http.post(
            "/v1/memory/recall",
            json={"query": query, "top_k": top_k, "as_of": as_of},
        )
        resp.raise_for_status()
        return list(resp.json()["facts"])

    async def consolidate(self, batch_size: int = 10) -> dict[str, Any]:
        if self.mode == "local":
            assert self._consolidator is not None
            rep: ConsolidationReport = await self._consolidator.run_once(batch_size=batch_size)
            return rep.to_dict()
        assert self._http is not None
        resp = await self._http.post("/v1/memory/consolidate", json={"batch_size": batch_size})
        resp.raise_for_status()
        return resp.json()

    async def health(self) -> dict[str, Any]:
        if self.mode == "local":
            assert self._engine is not None
            assess = self._engine.governor.evaluate(sample_interval=None)
            ollama = await self._engine.client.health()
            return {
                "governor": assess.to_dict(),
                "ollama": ollama.to_dict(),
                "has_vec": self._db.has_vec if self._db else False,
            }
        assert self._http is not None
        resp = await self._http.get("/health")
        resp.raise_for_status()
        return resp.json()
