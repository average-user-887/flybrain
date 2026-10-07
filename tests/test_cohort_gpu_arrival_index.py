"""The arrival-bitmap delivery index against the strict full-scan kernel (rc1).

Same device, same inputs: states (canonical and raw device arrays) and spikes
must be byte-identical, and the bitmap must be all zero between calls.  The
strict kernel is read from git (``v0.5.0rc1``); the test skips without it.
GPU only; ``NEUROFLY_COHORT_GPU`` selects the device.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')

from brainlab.cohort import contract as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def _gpu_ok() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(not _gpu_ok(), reason='needs a CUDA device with CuPy')


def _baseline():
    try:
        subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--verify', 'v0.5.0rc1^{commit}'],
                       check=True, capture_output=True)
    except Exception:
        pytest.skip('strict kernel tag v0.5.0rc1 not available')
    import cohort_profile
    return cohort_profile


def _graph(n=2000, k=24, seed=11):
    """Synthetic graph with duplicate edges, inhibition and hubs (high in-degree)."""
    rng = np.random.default_rng(seed)
    degree = rng.poisson(k, size=n)
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(degree, out=ptr[1:])
    m = int(ptr[-1])
    post = np.where(rng.random(m) < 0.2, rng.integers(0, 8, size=m), rng.integers(0, n, size=m)).astype(np.int32)
    dup = rng.random(m) < 0.1                     # repeat the previous edge's target (same pre mostly)
    post[1:][dup[1:]] = post[:-1][dup[1:]]
    weight = (rng.geometric(0.3, size=m) * np.where(rng.random(m) < 0.3, -1, 1) * 0.275 * 6).astype(np.float32)
    return dict(ptr=ptr, post=post, weight=np.ascontiguousarray(weight), ids=np.arange(n, dtype=np.int64))


@needs_gpu
@pytest.mark.parametrize('B', [1, 3])
def test_arrival_bitmap_is_byte_identical_to_the_strict_kernel(B):
    prof = _baseline()
    from brainlab.cohort import gpu as G
    base_mod = prof.baseline_module('v0.5.0rc1')
    arrays = _graph()
    dev = os.environ.get('NEUROFLY_COHORT_GPU')
    a = base_mod.GpuCohortEngine(arrays, B, device=dev, validate=False)
    b = G.GpuCohortEngine(arrays, B, device=dev, validate=False)
    assert b.describe()['delivery_index'] == G.DELIVERY_INDEX
    assert b.describe()['delivery'] == base_mod.DELIVERY
    drives = C.drive_phases(b.n, B)
    total = 0
    for t0, call in [(0, 1), (1, 7), (8, 20), (28, 40), (68, 60), (128, 72)]:
        d = drives[0 if t0 < 100 else 100]
        sa, sb = a.step(d, call), b.step(d, call)
        assert sa.tobytes() == sb.tobytes()
        total += int(sb.sum())
        assert prof._diff(a, b, f'call at {t0}') == []
        assert int(b.d_bits.sum()) == 0
    assert total > 0
    # Resume through write_state: both from the same saved state.
    saved = [a.read_state(k) for k in range(B)]
    for e in (a, b):
        for k in range(B):
            e.write_state(k, saved[k])
    for e in (a, b):
        e.step(drives[100], 50)
    assert prof._diff(a, b, 'resumed') == []
