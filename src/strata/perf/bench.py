"""Phase 0 diagnostics (`strata doctor`) and CPU thread sweep benchmarks."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from strata.config import StrataConfig
from strata.governor.hardware import (
    CpuTopology,
    OllamaEnvAudit,
    audit_ollama_env,
    probe_cpu_topology,
    probe_disk_free_gb,
    probe_memory,
)
from strata.governor.resource_governor import GovernorAssessment, ResourceGovernor
from strata.inference.client import OllamaClient, OllamaHealth


@dataclass(slots=True)
class DoctorCheck:
    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class DoctorReport:
    timestamp: str
    python_version: str
    platform: str
    cpu: CpuTopology
    governor: GovernorAssessment
    disk_free_gb: float
    ollama_env: OllamaEnvAudit
    ollama: OllamaHealth
    checks: list[DoctorCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(c.status == "FAIL" for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "python_version": self.python_version,
            "platform": self.platform,
            "cpu": self.cpu.to_dict(),
            "governor": self.governor.to_dict(),
            "disk_free_gb": self.disk_free_gb,
            "ollama_env": self.ollama_env.to_dict(),
            "ollama": self.ollama.to_dict(),
            "checks": [c.to_dict() for c in self.checks],
            "passed": self.passed,
        }


async def run_doctor(
    config: StrataConfig,
    *,
    client: OllamaClient | None = None,
) -> DoctorReport:
    """Run hardware, RAM budget, environment, and Ollama readiness checks."""
    cpu = probe_cpu_topology()
    governor = ResourceGovernor(config).evaluate(sample_interval=0.1)
    disk_free_gb = probe_disk_free_gb(".")
    ollama_env = audit_ollama_env()

    owns_client = client is None
    active_client = client or OllamaClient(config.ollama_host, timeout=10.0)
    try:
        ollama_health = await active_client.health()
    finally:
        if owns_client:
            await active_client.aclose()

    checks: list[DoctorCheck] = []

    py_ok = sys.version_info >= (3, 12)
    checks.append(
        DoctorCheck(
            name="python_version",
            status="PASS" if py_ok else "FAIL",
            detail=f"Python {sys.version.split()[0]} (requires >= 3.12)",
        )
    )

    core_desc = f"{cpu.physical_cores} physical / {cpu.logical_cpus} logical"
    if cpu.is_hybrid:
        core_desc += f" ({cpu.p_cores} P-cores, {cpu.e_cores} E-cores)"
    checks.append(
        DoctorCheck(
            name="cpu_topology",
            status="PASS",
            detail=f"{core_desc}; recommended num_thread={cpu.recommended_num_thread}",
        )
    )

    mem = governor.memory
    ram_status = "PASS" if governor.allow_new_requests else "FAIL"
    if governor.state.value == "yellow":
        ram_status = "WARN"
    checks.append(
        DoctorCheck(
            name="system_headroom",
            status=ram_status,
            detail=(
                f"Governor={governor.state.value.upper()} | "
                f"avail={mem.available_mb:.0f}/{mem.total_mb:.0f} MiB | "
                f"CPU={governor.cpu_percent:.1f}%"
            ),
        )
    )

    runtime_ok = mem.strata_rss_mb <= config.runtime_rss_ceiling_mb
    checks.append(
        DoctorCheck(
            name="strata_runtime_rss",
            status="PASS" if runtime_ok else "WARN",
            detail=f"{mem.strata_rss_mb:.1f} MiB (ceiling {config.runtime_rss_ceiling_mb:.0f} MiB)",
        )
    )

    stack_status = (
        "PASS"
        if governor.within_stack_target
        else ("WARN" if governor.within_stack_ceiling else "FAIL")
    )
    checks.append(
        DoctorCheck(
            name="ai_stack_rss_budget",
            status=stack_status,
            detail=(
                f"stack={mem.stack_rss_mb:.1f} MiB "
                f"(strata={mem.strata_rss_mb:.1f}, ollama={mem.ollama_rss_mb:.1f}; "
                f"target<={config.stack_rss_target_mb:.0f}, ceiling<={config.stack_rss_ceiling_mb:.0f})"
            ),
        )
    )

    checks.append(
        DoctorCheck(
            name="ollama_server",
            status="PASS" if ollama_health.reachable else "FAIL",
            detail=(
                f"v{ollama_health.version} at {ollama_health.host}"
                if ollama_health.reachable
                else f"Unreachable at {ollama_health.host}: {ollama_health.error}"
            ),
        )
    )

    if ollama_health.reachable:
        for role, model_tag in (
            ("primary_model", config.primary_model),
            ("router_model", config.router_model),
        ):
            found = ollama_health.has_model(model_tag)
            checks.append(
                DoctorCheck(
                    name=role,
                    status="PASS" if found else "WARN",
                    detail=(
                        f"{model_tag} installed"
                        if found
                        else f"{model_tag} not pulled (run: ollama pull {model_tag})"
                    ),
                )
            )

    checks.append(
        DoctorCheck(
            name="ollama_env_guardrails",
            status="PASS" if not ollama_env.warnings else "WARN",
            detail=(
                "All 8 GB Ollama env guardrails configured."
                if not ollama_env.warnings
                else "; ".join(ollama_env.warnings)
            ),
        )
    )

    return DoctorReport(
        timestamp=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        python_version=sys.version.split()[0],
        platform=sys.platform,
        cpu=cpu,
        governor=governor,
        disk_free_gb=disk_free_gb,
        ollama_env=ollama_env,
        ollama=ollama_health,
        checks=checks,
    )


@dataclass(slots=True)
class ThreadSweepPoint:
    num_thread: int
    cold_ttft_ms: float
    cold_load_ms: float
    warm_ttft_mean_ms: float
    warm_prompt_tok_s: float
    warm_eval_tok_s: float
    warm_wall_mean_ms: float
    peak_ollama_rss_mb: float
    peak_stack_rss_mb: float
    within_3gb_ceiling: bool
    trials: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ThreadSweepReport:
    timestamp: str
    model: str
    think: bool
    num_ctx: int
    max_tokens: int
    cpu: CpuTopology
    results: list[ThreadSweepPoint]
    recommended_num_thread: int
    saved_to: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "think": self.think,
            "num_ctx": self.num_ctx,
            "max_tokens": self.max_tokens,
            "cpu": self.cpu.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "recommended_num_thread": self.recommended_num_thread,
            "saved_to": self.saved_to,
        }


DEFAULT_BENCH_PROMPT = (
    "You are running inside a constrained local runtime. "
    "Summarize in exactly three short bullet points why prefix caching, "
    "schema validation, and bounded context windows keep small language models fast and reliable."
)


async def run_thread_sweep(
    config: StrataConfig,
    *,
    model: str | None = None,
    threads: list[int] | None = None,
    num_ctx: int | None = None,
    think: bool = False,
    trials: int = 2,
    max_tokens: int = 128,
    save: bool = False,
    results_dir: Path = Path("benchmarks/results"),
    client: OllamaClient | None = None,
) -> ThreadSweepReport:
    """Benchmark cold/warm TTFT, tok/s, and peak RSS across `num_thread` values."""
    target_model = model or config.primary_model
    ctx = num_ctx or config.num_ctx
    cpu = probe_cpu_topology()
    sweep_threads = threads or [2, 4, 6, 8, 12]
    # Deduplicate while preserving order and filtering invalid counts
    seen: set[int] = set()
    clean_threads: list[int] = []
    for t in sweep_threads:
        if 1 <= t <= max(cpu.logical_cpus, 1) and t not in seen:
            seen.add(t)
            clean_threads.append(t)
    if not clean_threads:
        clean_threads = [cpu.recommended_num_thread]

    owns_client = client is None
    active_client = client or OllamaClient(config.ollama_host, timeout=180.0)
    points: list[ThreadSweepPoint] = []
    messages = [{"role": "user", "content": DEFAULT_BENCH_PROMPT}]

    try:
        for n_threads in clean_threads:
            # Force clean model reload per thread count so runner applies num_thread.
            await active_client.unload_model(target_model)

            cold_res = await active_client.chat(
                target_model,
                messages,
                think=think,
                num_ctx=ctx,
                num_thread=n_threads,
                temperature=0.0,
                max_tokens=max_tokens,
                keep_alive=config.keep_alive,
                stream=True,
            )
            mem_snap = probe_memory(fresh=True)
            peak_ollama_rss = mem_snap.ollama_rss_mb
            peak_stack_rss = mem_snap.stack_rss_mb

            warm_ttfts: list[float] = []
            warm_prompt_tps: list[float] = []
            warm_eval_tps: list[float] = []
            warm_walls: list[float] = []

            for _ in range(max(1, trials)):
                res = await active_client.chat(
                    target_model,
                    messages,
                    think=think,
                    num_ctx=ctx,
                    num_thread=n_threads,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    keep_alive=config.keep_alive,
                    stream=True,
                )
                snap = probe_memory(fresh=True)
                peak_ollama_rss = max(peak_ollama_rss, snap.ollama_rss_mb)
                peak_stack_rss = max(peak_stack_rss, snap.stack_rss_mb)
                warm_ttfts.append(res.metrics.ttft_ms)
                warm_prompt_tps.append(res.metrics.prompt_tok_per_sec)
                warm_eval_tps.append(res.metrics.eval_tok_per_sec)
                warm_walls.append(res.metrics.wall_ms)

            points.append(
                ThreadSweepPoint(
                    num_thread=n_threads,
                    cold_ttft_ms=cold_res.metrics.ttft_ms,
                    cold_load_ms=cold_res.metrics.load_duration_ms,
                    warm_ttft_mean_ms=round(statistics.mean(warm_ttfts), 2),
                    warm_prompt_tok_s=round(statistics.mean(warm_prompt_tps), 2),
                    warm_eval_tok_s=round(statistics.mean(warm_eval_tps), 2),
                    warm_wall_mean_ms=round(statistics.mean(warm_walls), 2),
                    peak_ollama_rss_mb=round(peak_ollama_rss, 2),
                    peak_stack_rss_mb=round(peak_stack_rss, 2),
                    within_3gb_ceiling=peak_stack_rss <= config.stack_rss_ceiling_mb,
                    trials=max(1, trials),
                )
            )
    finally:
        if owns_client:
            await active_client.aclose()

    # Prefer highest throughput within 1% tolerance, breaking ties toward fewer threads
    # (avoids SMT/E-core saturation when physical cores already saturate memory bandwidth).
    max_eval = max(p.warm_eval_tok_s for p in points)
    near_peak = [p for p in points if p.warm_eval_tok_s >= max_eval * 0.99]
    best = min(near_peak, key=lambda p: (p.num_thread, p.warm_ttft_mean_ms))

    report = ThreadSweepReport(
        timestamp=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        model=target_model,
        think=think,
        num_ctx=ctx,
        max_tokens=max_tokens,
        cpu=cpu,
        results=points,
        recommended_num_thread=best.num_thread,
    )

    if save:
        results_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = target_model.replace(":", "_").replace("/", "_")
        out_path = results_dir / f"thread_sweep_{safe_model}_{stamp}.json"
        out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        report.saved_to = str(out_path)

    return report


@dataclass(slots=True)
class ColdStartReport:
    runs: int
    samples_ms: list[float]
    min_ms: float
    median_ms: float
    max_ms: float
    target_ms: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure_cold_start(runs: int = 5, target_ms: float = 300.0) -> ColdStartReport:
    """Measure fresh subprocess CLI startup latency (`import strata.cli; build_parser()`)."""
    samples: list[float] = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        subprocess.run(
            [sys.executable, "-c", "from strata.cli import build_parser; build_parser()"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        samples.append(round((time.perf_counter() - t0) * 1000.0, 2))

    med = round(statistics.median(samples), 2)
    return ColdStartReport(
        runs=len(samples),
        samples_ms=samples,
        min_ms=min(samples),
        median_ms=med,
        max_ms=max(samples),
        target_ms=target_ms,
        passed=med < target_ms,
    )
