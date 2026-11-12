"""Manifest-driven plugin and MCP adapter loading capability-tagged tools."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import BaseModel, Field

from strata.agents.policy import Capability
from strata.agents.runtime import RegisteredTool
from strata.agents.sandbox import run_sandboxed_subprocess
from strata.inference.tool_pruner import ToolSpec


class PluginToolDef(BaseModel):
    name: str
    description: str
    capability: Capability = Capability.READ
    tags: list[str] = Field(default_factory=list)
    always_include: bool = False
    returns_untrusted: bool = False
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )
    entrypoint: str | None = None  # module.path:callable
    command: list[str] | None = None  # sandboxed subprocess template e.g. ["python", "-c", "..."]
    timeout_sec: float = 10.0
    memory_limit_mb: int = 128


class PluginManifest(BaseModel):
    name: str
    version: str = "0.1.0"
    description: str = ""
    tools: list[PluginToolDef] = Field(default_factory=list)


def _load_python_entrypoint(ref: str) -> Callable[[dict[str, Any]], Any]:
    if ":" not in ref:
        raise ValueError(f"Invalid entrypoint {ref!r}; expected 'module:callable'")
    mod_name, attr_name = ref.split(":", 1)
    module = importlib.import_module(mod_name)
    fn = getattr(module, attr_name)
    if not callable(fn):
        raise TypeError(f"Entrypoint {ref!r} is not callable")
    return fn


def _build_sandboxed_command_handler(
    template: Sequence[str],
    *,
    cwd: Path | None,
    timeout_sec: float,
    memory_limit_mb: int,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    tpl = list(template)

    def _handler(args: dict[str, Any]) -> dict[str, Any]:
        safe_args: dict[str, str] = {}
        for k, v in args.items():
            s = str(v)
            if "\x00" in s:
                raise ValueError(f"Null byte rejected in argument {k!r}")
            safe_args[k] = s
        cmd = [part.format(**safe_args) for part in tpl]
        res = run_sandboxed_subprocess(
            cmd,
            cwd=cwd,
            timeout_sec=timeout_sec,
            memory_limit_mb=memory_limit_mb,
        )
        return {
            "ok": res.ok,
            "returncode": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "timed_out": res.timed_out,
        }

    return _handler


def load_manifest_file(
    path: Path | str, *, workspace_root: Path | None = None
) -> tuple[PluginManifest, list[RegisteredTool]]:
    """Parse a plugin JSON manifest into capability-tagged RegisteredTool instances."""
    manifest_path = Path(path)
    manifest = PluginManifest.model_validate(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    tools: list[RegisteredTool] = []
    for tdef in manifest.tools:
        if tdef.entrypoint:
            handler = _load_python_entrypoint(tdef.entrypoint)
        elif tdef.command:
            handler = _build_sandboxed_command_handler(
                tdef.command,
                cwd=workspace_root,
                timeout_sec=tdef.timeout_sec,
                memory_limit_mb=tdef.memory_limit_mb,
            )
        else:
            raise ValueError(
                f"Plugin tool {tdef.name!r} in {manifest.name!r} needs entrypoint or command"
            )

        spec = ToolSpec(
            name=tdef.name,
            description=tdef.description,
            parameters=tdef.parameters,
            tags=tuple(tdef.tags),
            always_include=tdef.always_include,
            capability=tdef.capability.value,
        )
        tools.append(
            RegisteredTool(
                spec=spec,
                capability=tdef.capability,
                handler=handler,
                returns_untrusted=tdef.returns_untrusted,
            )
        )
    return manifest, tools


def adapt_mcp_tool(
    mcp_schema: dict[str, Any],
    executor: Callable[[str, dict[str, Any]], Any],
    *,
    capability: Capability = Capability.NETWORK,
    returns_untrusted: bool = True,
) -> RegisteredTool:
    """Adapt an MCP tool JSON descriptor into a CaMeL taint-tagged RegisteredTool."""
    name = str(mcp_schema["name"])
    description = str(mcp_schema.get("description") or "")
    parameters = mcp_schema.get("inputSchema") or {
        "type": "object",
        "properties": {},
        "required": [],
    }
    spec = ToolSpec(
        name=name,
        description=description,
        parameters=parameters,
        tags=("mcp",),
        capability=capability.value,
    )

    def _call(args: dict[str, Any]) -> Any:
        return executor(name, args)

    return RegisteredTool(
        spec=spec,
        capability=capability,
        handler=_call,
        returns_untrusted=returns_untrusted,
    )


class PluginRegistry:
    """Discovers manifest plugins from disk and collects RegisteredTool entries."""

    def __init__(self, *, workspace_root: Path | None = None) -> None:
        self.workspace_root = workspace_root
        self.manifests: dict[str, PluginManifest] = {}
        self.tools: dict[str, RegisteredTool] = {}

    def load_directory(self, plugins_dir: Path | str) -> int:
        directory = Path(plugins_dir)
        if not directory.exists():
            return 0
        loaded = 0
        for mf_path in sorted(directory.glob("*.json")):
            manifest, tools = load_manifest_file(
                mf_path, workspace_root=self.workspace_root
            )
            self.manifests[manifest.name] = manifest
            for tool in tools:
                self.tools[tool.spec.name] = tool
            loaded += 1
        return loaded

    def register_tool(self, tool: RegisteredTool) -> None:
        self.tools[tool.spec.name] = tool

    def list_tools(self) -> list[RegisteredTool]:
        return [self.tools[name] for name in sorted(self.tools.keys())]
