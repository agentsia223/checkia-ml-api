"""Runtime tuning: size CPU thread pools to the container's CPU quota.

In a container, the native math libraries behind torch and numpy (OpenMP / MKL /
OpenBLAS) detect the *host* core count and start that many threads. When the
cgroup CPU quota is smaller than the host core count, this oversubscribes the
CPU and triggers CFS throttling, which severely degrades inference latency.

We read the quota and cap the pools to match via the standard thread env vars,
set *before* those libraries initialise. (Setting torch's thread counts through
its Python API at runtime — after import, before model load — can deadlock, so
we use the environment instead, which is what the library authors expect.)
"""

from __future__ import annotations

import math
import os
from pathlib import Path

_CGROUP_V2 = Path("/sys/fs/cgroup/cpu.max")
_CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
_CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


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


def configure_thread_env() -> int:
    """Cap CPU-library thread pools to the cgroup quota via env vars.

    Must run before torch / numpy import their native threading libraries. Uses
    ``setdefault`` so an explicit override (e.g. a Railway env var) still wins.
    Returns the resolved thread count.
    """
    n = resolve_thread_count(cgroup_cpu_quota(), os.cpu_count())
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, str(n))
    return n
