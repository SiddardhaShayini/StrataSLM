"""Three-stage Resource Governor (GREEN / YELLOW / RED) protecting 8 GB RAM."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import psutil

from strata.config import StrataConfig
from strata.governor.hardware import MemorySnapshot, probe_memory

# Process-wide psutil.cpu_percent(interval=None) delta state cannot safely be polled
# faster than ~0.5s without measuring single-quantum 100% OS scheduling spikes.
psutil.cpu_percent(interval=None)
_LAST_CPU_TS: float = time.monotonic()
_LAST_CPU_PCT: float = 0.0


def _sample_cpu_percent(sample_interval: float | None, *, min_window_sec: float = 0.5) -> float:
    global _LAST_CPU_TS, _LAST_CPU_PCT
    if sample_interval is not None:
        val = float(psutil.cpu_percent(interval=sample_interval))
        _LAST_CPU_TS = time.monotonic()
        _LAST_CPU_PCT = val
        return val
    now = time.monotonic()
    if (now - _LAST_CPU_TS) >= min_window_sec:
        _LAST_CPU_PCT = float(psutil.cpu_percent(interval=None))
        _LAST_CPU_TS = now
    return _LAST_CPU_PCT


class GovernorState(StrEnum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass(slots=True)
class GovernorAssessment:
    """Result of evaluating RAM, CPU, and AI stack RSS against 8 GB limits."""

    state: GovernorState
    cpu_percent: float
    memory: MemorySnapshot
    reasons: list[str]
    allow_sleep_jobs: bool
    should_trim_caches: bool
    allow_new_requests: bool
    within_stack_target: bool
    within_stack_ceiling: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "cpu_percent": self.cpu_percent,
            "memory": self.memory.to_dict(),
            "reasons": list(self.reasons),
            "allow_sleep_jobs": self.allow_sleep_jobs,
            "should_trim_caches": self.should_trim_caches,
            "allow_new_requests": self.allow_new_requests,
            "within_stack_target": self.within_stack_target,
            "within_stack_ceiling": self.within_stack_ceiling,
        }


class ResourceGovernor:
    """Evaluates system pressure and enforces back-pressure on background/heavy tasks."""

    def __init__(self, config: StrataConfig) -> None:
        self._config = config

    def evaluate(
        self,
        *,
        sample_interval: float | None = None,
        memory: MemorySnapshot | None = None,
        cpu_percent: float | None = None,
    ) -> GovernorAssessment:
        """Assess current system pressure. Pass `memory`/`cpu_percent` to test deterministically."""
        mem = memory or probe_memory()
        cpu = (
            cpu_percent
            if cpu_percent is not None
            else _sample_cpu_percent(sample_interval)
        )

        reasons: list[str] = []
        state = GovernorState.GREEN

        # Check RED thresholds first
        if mem.available_mb < self._config.ram_red_avail_mb:
            state = GovernorState.RED
            reasons.append(
                f"Available RAM {mem.available_mb:.0f} MiB < red limit {self._config.ram_red_avail_mb:.0f} MiB"
            )
        if cpu > self._config.cpu_red_pct:
            state = GovernorState.RED
            reasons.append(
                f"CPU usage {cpu:.1f}% > red limit {self._config.cpu_red_pct:.1f}%"
            )

        # Check YELLOW thresholds if not already RED
        if state == GovernorState.GREEN:
            if mem.available_mb < self._config.ram_yellow_avail_mb:
                state = GovernorState.YELLOW
                reasons.append(
                    f"Available RAM {mem.available_mb:.0f} MiB < yellow limit {self._config.ram_yellow_avail_mb:.0f} MiB"
                )
            if cpu >= self._config.cpu_yellow_pct:
                state = GovernorState.YELLOW
                reasons.append(
                    f"CPU usage {cpu:.1f}% >= yellow limit {self._config.cpu_yellow_pct:.1f}%"
                )

        stack_rss = mem.stack_rss_mb
        within_target = stack_rss <= self._config.stack_rss_target_mb
        within_ceiling = stack_rss <= self._config.stack_rss_ceiling_mb
        if not within_ceiling:
            reasons.append(
                f"AI stack RSS {stack_rss:.0f} MiB exceeds hard ceiling {self._config.stack_rss_ceiling_mb:.0f} MiB"
            )
            if state == GovernorState.GREEN:
                state = GovernorState.YELLOW

        if mem.strata_rss_mb > self._config.runtime_rss_ceiling_mb:
            reasons.append(
                f"Strata runtime RSS {mem.strata_rss_mb:.1f} MiB exceeds {self._config.runtime_rss_ceiling_mb:.0f} MiB ceiling"
            )

        if not reasons:
            reasons.append("Headroom healthy for primary inference and background consolidation.")

        return GovernorAssessment(
            state=state,
            cpu_percent=round(cpu, 1),
            memory=mem,
            reasons=reasons,
            allow_sleep_jobs=(state == GovernorState.GREEN),
            should_trim_caches=(state in {GovernorState.YELLOW, GovernorState.RED}),
            allow_new_requests=(state != GovernorState.RED),
            within_stack_target=within_target,
            within_stack_ceiling=within_ceiling,
        )
