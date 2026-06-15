"""Runtime tuning: size CPU thread pools to the container's CPU quota.

In a container, PyTorch detects the *host* core count and spawns that many
intra-op threads. When the cgroup CPU quota is smaller than the host core count,
this oversubscribes the CPU and triggers CFS throttling, which severely degrades
inference latency. We read the quota and cap the thread pools to match.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

_CGROUP_V2 = Path("/sys/fs/cgroup/cpu.max")
_CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
_CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def cgroup_cpu_quota(
    v2: Path = _CGROUP_V2,
    v1_quota: Path = _CGROUP_V1_QUOTA,
    v1_period: Path = _CGROUP_V1_PERIOD,
) -> float | None:
    """CPU cores the cgroup permits, or None if unlimited or unavailable.

    Supports cgroup v2 (``cpu.max`` = ``"<quota> <period>"`` or ``"max ..."``)
    and v1 (``cpu.cfs_quota_us`` / ``cpu.cfs_period_us``, quota ``-1`` = unlimited).
    """
    raw = _read(v2)
    if raw is not None:
        parts = raw.split()
        if len(parts) == 2 and parts[0] != "max":
            quota, period = float(parts[0]), float(parts[1])
            if quota > 0 and period > 0:
                return quota / period
        return None  # "max <period>" => unlimited

    quota_raw, period_raw = _read(v1_quota), _read(v1_period)
    if quota_raw is not None and period_raw is not None:
        quota, period = float(quota_raw), float(period_raw)
        if quota > 0 and period > 0:
            return quota / period
    return None


def resolve_thread_count(quota: float | None, cpu_count: int | None) -> int:
    """Threads to use: the quota floored to whole cores, clamped to [1, cpu_count]."""
    cores = cpu_count or 1
    if quota is None:
        return max(1, cores)
    return max(1, min(int(math.floor(quota)), cores))


def configure_torch_threads(logger=None) -> int:
    """Cap torch intra-op/inter-op thread pools to the cgroup CPU quota.

    Returns the thread count applied. Call once at startup, before models load.
    """
    import torch

    n = resolve_thread_count(cgroup_cpu_quota(), os.cpu_count())
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:
        # The interop pool is already initialised; it can only be set once.
        pass
    if logger is not None:
        logger.info("Capped torch threads to %d (cgroup-aware)", n)
    return n
