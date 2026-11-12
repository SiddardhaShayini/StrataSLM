"""Resource-capped subprocess sandbox using Win32 Job Objects on Windows."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(slots=True)
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: float
    memory_limit_mb: int

    @property
    def ok(self) -> bool:
        return not self.timed_out and self.returncode == 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _create_windows_job(memory_limit_mb: int) -> int | None:
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None

    limit_kill_on_close = 0x2000
    limit_process_memory = 0x0100
    limit_job_memory = 0x0200
    limit_bytes = max(16, memory_limit_mb) * 1024 * 1024

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = (
        limit_kill_on_close | limit_process_memory | limit_job_memory
    )
    info.ProcessMemoryLimit = limit_bytes
    info.JobMemoryLimit = limit_bytes

    ok = kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
    if not ok:
        kernel32.CloseHandle(job)
        return None
    return int(job)


def _assign_windows_job(job_handle: int, pid: int) -> None:
    if sys.platform != "win32":
        return
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    process_all_access = 0x1F0FFF
    proc_handle = kernel32.OpenProcess(process_all_access, False, int(pid))
    if proc_handle:
        try:
            kernel32.AssignProcessToJobObject(job_handle, proc_handle)
        finally:
            kernel32.CloseHandle(proc_handle)


def _close_windows_job(job_handle: int | None) -> None:
    if job_handle is None or sys.platform != "win32":
        return
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle(job_handle)


def run_sandboxed_subprocess(
    command: Sequence[str],
    *,
    cwd: Path | str | None = None,
    timeout_sec: float = 10.0,
    memory_limit_mb: int = 128,
    max_output_chars: int = 8192,
) -> SandboxResult:
    """Execute a subprocess capped by timeout, bounded output, and Win32 Job memory limits."""
    t0 = time.perf_counter()
    job_handle = _create_windows_job(memory_limit_mb)
    timed_out = False

    try:
        proc = subprocess.Popen(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if job_handle is not None:
            _assign_windows_job(job_handle, proc.pid)

        try:
            out, err = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            out, err = proc.communicate()

        return SandboxResult(
            returncode=int(proc.returncode if proc.returncode is not None else -1),
            stdout=out[:max_output_chars],
            stderr=err[:max_output_chars],
            timed_out=timed_out,
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            memory_limit_mb=memory_limit_mb,
        )
    finally:
        _close_windows_job(job_handle)
