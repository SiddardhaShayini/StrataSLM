"""Cross-platform + Win32 hardware topology, RAM, and Ollama process telemetry.

Avoids spawning `wmic`, `systeminfo`, or `Get-CimInstance` subprocesses (which are
slow or restricted in Windows environments). Uses `ctypes` Win32 APIs + `psutil`.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil

_SELF_PROC = psutil.Process(os.getpid())
_OLLAMA_PID_CACHE: tuple[float, list[int]] = (0.0, [])


@dataclass(slots=True)
class CpuTopology:
    """Detected CPU core layout including Intel hybrid P-core / E-core distinction."""

    arch: str
    processor: str
    logical_cpus: int
    physical_cores: int
    p_cores: int | None
    e_cores: int | None
    is_hybrid: bool
    recommended_num_thread: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemorySnapshot:
    """Point-in-time system and AI stack RSS telemetry in megabytes (MiB)."""

    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float
    strata_rss_mb: float
    ollama_rss_mb: float
    ollama_pids: list[int]

    @property
    def stack_rss_mb(self) -> float:
        return round(self.strata_rss_mb + self.ollama_rss_mb, 2)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stack_rss_mb"] = self.stack_rss_mb
        return data


@dataclass(slots=True)
class OllamaEnvAudit:
    """Audit of Ollama server environment variables against 8 GB RAM rules."""

    max_loaded_models: str | None
    num_parallel: str | None
    kv_cache_type: str | None
    flash_attention: str | None
    ollama_host: str | None
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _probe_windows_hybrid_cores() -> tuple[int, int | None, int | None, bool] | None:
    """Use Win32 GetLogicalProcessorInformationEx to inspect physical & hybrid cores."""
    if sys.platform != "win32":
        return None

    import ctypes
    from ctypes import wintypes

    relation_processor_core = 0
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class GROUP_AFFINITY(ctypes.Structure):
        _fields_ = [
            ("Mask", ctypes.c_size_t),
            ("Group", wintypes.WORD),
            ("Reserved", wintypes.WORD * 3),
        ]

    class PROCESSOR_RELATIONSHIP(ctypes.Structure):
        _fields_ = [
            ("Flags", ctypes.c_ubyte),
            ("EfficiencyClass", ctypes.c_ubyte),
            ("Reserved", ctypes.c_ubyte * 20),
            ("GroupCount", wintypes.WORD),
            ("GroupMask", GROUP_AFFINITY * 1),
        ]

    class SYSTEM_LOGICAL_PROCESSOR_INFO_EX(ctypes.Structure):
        _fields_ = [
            ("Relationship", ctypes.c_int),
            ("Size", ctypes.c_uint32),
            ("Processor", PROCESSOR_RELATIONSHIP),
        ]

    buf_len = wintypes.DWORD(0)
    kernel32.GetLogicalProcessorInformationEx(
        relation_processor_core, None, ctypes.byref(buf_len)
    )
    if buf_len.value == 0:
        return None

    buf = ctypes.create_string_buffer(buf_len.value)
    ok = kernel32.GetLogicalProcessorInformationEx(
        relation_processor_core, buf, ctypes.byref(buf_len)
    )
    if not ok:
        return None

    eff_classes: list[int] = []
    offset = 0
    while offset < buf_len.value:
        entry = ctypes.cast(
            ctypes.byref(buf, offset),
            ctypes.POINTER(SYSTEM_LOGICAL_PROCESSOR_INFO_EX),
        ).contents
        if entry.Size == 0:
            break
        if entry.Relationship == relation_processor_core:
            eff_classes.append(int(entry.Processor.EfficiencyClass))
        offset += entry.Size

    if not eff_classes:
        return None

    physical = len(eff_classes)
    max_eff = max(eff_classes)
    min_eff = min(eff_classes)
    if max_eff > min_eff:
        p_cores = sum(1 for e in eff_classes if e == max_eff)
        e_cores = physical - p_cores
        return physical, p_cores, e_cores, True
    return physical, None, None, False


def _recommend_threads(
    logical_cpus: int, physical_cores: int, p_cores: int | None
) -> int:
    """Pick safe default thread count (PLAN.md §2: 4-6 physical P-cores)."""
    if p_cores is not None and p_cores > 0:
        return max(2, min(p_cores, 6))
    return max(2, min(physical_cores, 6, logical_cpus))


def probe_cpu_topology() -> CpuTopology:
    """Detect logical CPUs, physical cores, and Intel hybrid P/E topology."""
    logical = os.cpu_count() or 1
    win_info = _probe_windows_hybrid_cores()
    if win_info is not None:
        physical, p_cores, e_cores, is_hybrid = win_info
    else:
        physical = psutil.cpu_count(logical=False) or logical
        p_cores = None
        e_cores = None
        is_hybrid = False

    return CpuTopology(
        arch=platform.machine() or "unknown",
        processor=platform.processor() or "unknown",
        logical_cpus=logical,
        physical_cores=physical,
        p_cores=p_cores,
        e_cores=e_cores,
        is_hybrid=is_hybrid,
        recommended_num_thread=_recommend_threads(logical, physical, p_cores),
    )


def _is_ollama_process(name: str | None) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return (
        lowered.startswith("ollama")
        or "ollama_llama_server" in lowered
        or lowered in {"llama-server", "llama-server.exe"}
    )


def probe_memory(*, fresh: bool = False, ollama_scan_ttl: float = 5.0) -> MemorySnapshot:
    """Measure system RAM plus current StrataSLM and Ollama daemon/runner RSS."""
    global _OLLAMA_PID_CACHE
    vm = psutil.virtual_memory()
    mib = 1024.0 * 1024.0

    strata_rss_mb = _SELF_PROC.memory_info().rss / mib

    now = time.monotonic()
    last_scan, cached_pids = _OLLAMA_PID_CACHE
    need_full_scan = fresh or (now - last_scan) >= ollama_scan_ttl

    ollama_rss_bytes = 0
    ollama_pids: list[int] = []
    if not need_full_scan:
        if cached_pids:
            for pid in cached_pids:
                try:
                    proc = psutil.Process(pid)
                    ollama_rss_bytes += proc.memory_info().rss
                    ollama_pids.append(pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    need_full_scan = True
                    break

    if need_full_scan:
        ollama_rss_bytes = 0
        ollama_pids = []
        for proc in psutil.process_iter(["pid", "name", "memory_info"]):
            try:
                info = proc.info
                if _is_ollama_process(info.get("name")):
                    mem = info.get("memory_info")
                    if mem is not None:
                        ollama_rss_bytes += mem.rss
                    ollama_pids.append(int(info["pid"]))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        ollama_pids = sorted(ollama_pids)
        _OLLAMA_PID_CACHE = (now, ollama_pids)

    return MemorySnapshot(
        total_mb=round(vm.total / mib, 2),
        available_mb=round(vm.available / mib, 2),
        used_mb=round(vm.used / mib, 2),
        percent_used=round(float(vm.percent), 1),
        strata_rss_mb=round(strata_rss_mb, 2),
        ollama_rss_mb=round(ollama_rss_bytes / mib, 2),
        ollama_pids=ollama_pids,
    )


def probe_disk_free_gb(path: Path | str = ".") -> float:
    """Return free disk space in GiB for the target path."""
    usage = shutil.disk_usage(Path(path).resolve())
    return round(usage.free / (1024.0**3), 2)


def audit_ollama_env() -> OllamaEnvAudit:
    """Check current environment variables for 8 GB Ollama safety settings."""
    max_loaded = os.environ.get("OLLAMA_MAX_LOADED_MODELS")
    num_parallel = os.environ.get("OLLAMA_NUM_PARALLEL")
    kv_cache = os.environ.get("OLLAMA_KV_CACHE_TYPE")
    flash_attn = os.environ.get("OLLAMA_FLASH_ATTENTION")
    ollama_host = os.environ.get("OLLAMA_HOST")

    warnings: list[str] = []
    if max_loaded != "1":
        warnings.append(
            f"OLLAMA_MAX_LOADED_MODELS is {max_loaded!r}; set to '1' to prevent model swap thrashing on 8 GB RAM."
        )
    if num_parallel != "1":
        warnings.append(
            f"OLLAMA_NUM_PARALLEL is {num_parallel!r}; set to '1' to avoid duplicate KV cache allocations."
        )
    if kv_cache not in {"q8_0", "q4_0"}:
        warnings.append(
            f"OLLAMA_KV_CACHE_TYPE is {kv_cache!r}; recommend 'q8_0' (with OLLAMA_FLASH_ATTENTION=1) to cut KV cache RAM."
        )
    if flash_attn not in {"1", "true", "TRUE"}:
        warnings.append(
            f"OLLAMA_FLASH_ATTENTION is {flash_attn!r}; recommend '1' so quantized KV cache activates cleanly."
        )

    return OllamaEnvAudit(
        max_loaded_models=max_loaded,
        num_parallel=num_parallel,
        kv_cache_type=kv_cache,
        flash_attention=flash_attn,
        ollama_host=ollama_host,
        warnings=warnings,
    )
