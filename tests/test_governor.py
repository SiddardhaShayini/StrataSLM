"""Tests for hardware probing and 3-stage Resource Governor."""

from strata.config import StrataConfig
from strata.governor.hardware import (
    MemorySnapshot,
    audit_ollama_env,
    probe_cpu_topology,
    probe_memory,
)
from strata.governor.resource_governor import GovernorState, ResourceGovernor


def test_probe_cpu_topology_sanity() -> None:
    topo = probe_cpu_topology()
    assert topo.logical_cpus >= 1
    assert topo.physical_cores >= 1
    assert topo.physical_cores <= topo.logical_cpus
    assert 2 <= topo.recommended_num_thread <= 6 or topo.logical_cpus < 2


def test_probe_memory_sanity() -> None:
    snap = probe_memory()
    assert snap.total_mb > 0
    assert snap.available_mb > 0
    assert snap.strata_rss_mb > 0
    assert snap.stack_rss_mb >= snap.strata_rss_mb


def test_audit_ollama_env(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_MAX_LOADED_MODELS", "1")
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "1")
    monkeypatch.setenv("OLLAMA_KV_CACHE_TYPE", "q8_0")
    monkeypatch.setenv("OLLAMA_FLASH_ATTENTION", "1")

    audit = audit_ollama_env()
    assert audit.warnings == []

    monkeypatch.delenv("OLLAMA_KV_CACHE_TYPE")
    audit_warn = audit_ollama_env()
    assert any("OLLAMA_KV_CACHE_TYPE" in w for w in audit_warn.warnings)


def _snap(*, avail_mb: float, strata_mb: float = 80.0, ollama_mb: float = 1800.0) -> MemorySnapshot:
    return MemorySnapshot(
        total_mb=8192.0,
        available_mb=avail_mb,
        used_mb=8192.0 - avail_mb,
        percent_used=round((8192.0 - avail_mb) / 8192.0 * 100.0, 1),
        strata_rss_mb=strata_mb,
        ollama_rss_mb=ollama_mb,
        ollama_pids=[1234] if ollama_mb > 0 else [],
    )


def test_governor_green_yellow_red() -> None:
    cfg = StrataConfig()
    gov = ResourceGovernor(cfg)

    green = gov.evaluate(memory=_snap(avail_mb=2400.0), cpu_percent=25.0)
    assert green.state == GovernorState.GREEN
    assert green.allow_sleep_jobs is True
    assert green.should_trim_caches is False
    assert green.allow_new_requests is True
    assert green.within_stack_target is True

    yellow_ram = gov.evaluate(memory=_snap(avail_mb=1500.0), cpu_percent=25.0)
    assert yellow_ram.state == GovernorState.YELLOW
    assert yellow_ram.allow_sleep_jobs is False
    assert yellow_ram.should_trim_caches is True
    assert yellow_ram.allow_new_requests is True

    yellow_cpu = gov.evaluate(memory=_snap(avail_mb=2500.0), cpu_percent=78.0)
    assert yellow_cpu.state == GovernorState.YELLOW

    red_ram = gov.evaluate(memory=_snap(avail_mb=1000.0), cpu_percent=25.0)
    assert red_ram.state == GovernorState.RED
    assert red_ram.allow_sleep_jobs is False
    assert red_ram.allow_new_requests is False

    red_cpu = gov.evaluate(memory=_snap(avail_mb=2500.0), cpu_percent=85.0)
    assert red_cpu.state == GovernorState.RED
    assert red_cpu.allow_new_requests is False

    over_ceiling = gov.evaluate(
        memory=_snap(avail_mb=2500.0, strata_mb=100.0, ollama_mb=3100.0),
        cpu_percent=20.0,
    )
    assert over_ceiling.within_stack_ceiling is False
    assert over_ceiling.state == GovernorState.YELLOW
