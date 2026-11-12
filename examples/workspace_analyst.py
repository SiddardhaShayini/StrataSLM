"""Sample App 1: Workspace Analyst — safe local file inspection via @tool + StrataClient."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from strata.agents.policy import Capability
from strata.config import StrataConfig
from strata.sdk import StrataClient, tool

WORKSPACE = Path(__file__).resolve().parents[1]


@tool(tags=("workspace", "files"), capability=Capability.READ)
def list_workspace_files(subdir: str = ".") -> list[str]:
    """List top-level files inside a workspace subdirectory."""
    target = (WORKSPACE / subdir).resolve()
    if target != WORKSPACE and WORKSPACE not in target.parents:
        raise PermissionError("Path escapes workspace")
    return sorted(p.name for p in target.iterdir() if not p.name.startswith("."))[:30]


@tool(tags=("workspace", "read"), capability=Capability.READ)
def read_snippet(path: str, max_lines: int = 25) -> str:
    """Read the first N lines of a text file inside the workspace."""
    target = (WORKSPACE / path).resolve()
    if target != WORKSPACE and WORKSPACE not in target.parents:
        raise PermissionError("Path escapes workspace")
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[:max_lines])


async def main() -> None:
    cfg = StrataConfig.from_env()
    async with StrataClient.local(
        cfg,
        tools=[list_workspace_files, read_snippet],
        workspace_root=WORKSPACE,
    ) as client:
        res = await client.run_agent(
            "List top-level files in src/strata and summarize the project layout."
        )
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
