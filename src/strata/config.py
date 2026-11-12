"""Configuration for StrataSLM runtime and 8 GB hardware guardrails."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def normalize_ollama_host(raw: str | None) -> str:
    """Normalize OLLAMA_HOST into a base HTTP URL without trailing slash."""
    host = (raw or "http://127.0.0.1:11434").strip()
    if not host:
        return "http://127.0.0.1:11434"
    if "://" not in host:
        host = f"http://{host}"
    host = host.rstrip("/")
    # If OLLAMA_HOST is set to e.g. 0.0.0.0:11434, loopback is safer for local clients.
    if host.startswith("http://0.0.0.0"):
        host = "http://127.0.0.1" + host[len("http://0.0.0.0") :]
    return host


@dataclass(slots=True)
class StrataConfig:
    """Core runtime settings tuned for an 8 GB RAM, CPU-only Intel i5 laptop."""

    ollama_host: str = "http://127.0.0.1:11434"
    primary_model: str = "qwen3:1.7b"
    router_model: str = "qwen3:0.6b"
    num_ctx: int = 4096
    num_thread: int = 4
    keep_alive: str = "5m"
    data_dir: Path = Path(".strata")

    # 3-stage Resource Governor thresholds (aligned with PLAN.md §2 & §7)
    ram_yellow_avail_mb: float = 1800.0
    ram_red_avail_mb: float = 1200.0
    cpu_yellow_pct: float = 75.0
    cpu_red_pct: float = 80.0

    # AI Stack RSS limits (MB)
    stack_rss_target_mb: float = 2560.0
    stack_rss_ceiling_mb: float = 3072.0
    runtime_rss_ceiling_mb: float = 180.0

    @property
    def db_path(self) -> Path:
        return self.data_dir / "strata.db"

    @classmethod
    def from_env(cls, ollama_host: str | None = None) -> StrataConfig:
        """Build configuration from environment variables with safe 8 GB defaults."""
        host = normalize_ollama_host(ollama_host or os.environ.get("OLLAMA_HOST"))
        num_ctx = int(os.environ.get("STRATA_NUM_CTX", "4096"))
        num_thread = int(os.environ.get("STRATA_NUM_THREAD", "4"))
        primary_model = os.environ.get("STRATA_PRIMARY_MODEL", "qwen3:1.7b")
        router_model = os.environ.get("STRATA_ROUTER_MODEL", "qwen3:0.6b")
        data_dir = Path(os.environ.get("STRATA_DATA_DIR", ".strata"))
        return cls(
            ollama_host=host,
            primary_model=primary_model,
            router_model=router_model,
            num_ctx=num_ctx,
            num_thread=num_thread,
            data_dir=data_dir,
        )
