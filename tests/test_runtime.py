from pathlib import Path

import pytest

from app.runtime import cgroup_cpu_quota, resolve_thread_count


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_cgroup_v2_limited(tmp_path):
    v2 = _write(tmp_path / "cpu.max", "2400000 100000")
    missing = tmp_path / "missing"
    assert cgroup_cpu_quota(v2, missing, missing) == 24.0


def test_cgroup_v2_unlimited(tmp_path):
    v2 = _write(tmp_path / "cpu.max", "max 100000")
    missing = tmp_path / "missing"
    assert cgroup_cpu_quota(v2, missing, missing) is None


def test_cgroup_v1_limited(tmp_path):
    missing = tmp_path / "no_v2"
    quota = _write(tmp_path / "quota", "200000")
    period = _write(tmp_path / "period", "100000")
    assert cgroup_cpu_quota(missing, quota, period) == 2.0


def test_cgroup_v1_unlimited(tmp_path):
    missing = tmp_path / "no_v2"
    quota = _write(tmp_path / "quota", "-1")
    period = _write(tmp_path / "period", "100000")
    assert cgroup_cpu_quota(missing, quota, period) is None


def test_cgroup_all_missing(tmp_path):
    missing = tmp_path / "missing"
    assert cgroup_cpu_quota(missing, missing, missing) is None


@pytest.mark.parametrize(
    "quota,cpu,expected",
    [
        (24.0, 48, 24),    # quota below host cores -> use quota
        (None, 8, 8),      # unlimited -> use cpu_count
        (100.0, 8, 8),     # quota above cpu_count -> clamp
        (2.9, 16, 2),      # fractional -> floor
        (0.5, 4, 1),       # sub-core -> at least 1
        (None, None, 1),   # nothing known -> 1
    ],
)
def test_resolve_thread_count(quota, cpu, expected):
    assert resolve_thread_count(quota, cpu) == expected


def test_configure_torch_threads_applies_to_torch():
    import torch

    from app.runtime import configure_torch_threads

    n = configure_torch_threads()
    assert isinstance(n, int) and n >= 1
    assert torch.get_num_threads() == n
