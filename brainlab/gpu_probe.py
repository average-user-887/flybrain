"""Decide whether a GPU build can really run, by running it once.

"A CUDA device is visible" is not enough.  On a fresh install:

* numba's ``cuda.is_available()`` can be True while nothing compiles
  (numba-cuda 0.30.4 calls ``np.row_stack``, which numpy 2.5 removed), or False
  because libNVVM is missing;
* CuPy can see the device but fail at the first kernel compile (no CUDA headers
  or NVRTC), or at the first sparse product (no ``libcusparse.so.12``), which the
  v4/v5 engines need.

Each probe here compiles and launches a tiny kernel of the same kind the engine
uses and checks the answer.  The result and a human-readable reason are cached
for the process.  When a library and a device are present but the probe fails,
the reason is logged as a WARNING (the brain then runs on the CPU), so a GPU
user is told why instead of silently getting the CPU.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

log = logging.getLogger('brainlab')

_RESULTS: Dict[str, Tuple[bool, str]] = {}


class _Absent(Exception):
    """No library or no device: an expected CPU-only host, not worth a warning."""


def _run(name: str, probe) -> Tuple[bool, str]:
    if name not in _RESULTS:
        try:
            detail = probe()
            _RESULTS[name] = (True, detail or 'ok')
        except _Absent as exc:
            _RESULTS[name] = (False, str(exc))
        except Exception as exc:  # a broken build: compile, launch or library load failed
            reason = f'{type(exc).__name__}: {exc}'.splitlines()[0][:300]
            _RESULTS[name] = (False, reason)
            log.warning('brainlab: the %s GPU build is unusable and will not be used (%s)', name, reason)
    return _RESULTS[name]


def _numba_cuda() -> str:
    try:
        from numba import cuda
    except Exception as exc:
        raise _Absent(f'numba.cuda not importable ({type(exc).__name__})')
    try:
        available = bool(cuda.is_available())
    except Exception as exc:
        raise _Absent(f'numba.cuda unavailable ({type(exc).__name__}: {exc})')
    if not available:
        raise _Absent('numba.cuda.is_available() is False (no device, or libNVVM missing)')
    import numpy as np

    @cuda.jit
    def _probe_kernel(out):
        i = cuda.grid(1)
        if i < out.shape[0]:
            out[i] = i + 1

    host = np.zeros(32, dtype=np.int32)
    device = cuda.to_device(host)
    _probe_kernel[1, 32](device)
    result = device.copy_to_host()
    if not np.array_equal(result, np.arange(1, 33, dtype=np.int32)):
        raise RuntimeError('numba.cuda probe kernel returned wrong values')
    return 'numba.cuda kernel compiled and ran'


_COOP_SOURCE = r'''
#include <cooperative_groups.h>
extern "C" __global__ void neurofly_probe(int* out) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = i + 1;
    grid.sync();
}
'''


def _cupy_device():
    try:
        import cupy
    except Exception as exc:
        raise _Absent(f'CuPy not installed ({type(exc).__name__}); install the gpu extra')
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:
        raise _Absent(f'no CUDA device visible to CuPy ({type(exc).__name__})')
    if count < 1:
        raise _Absent('no CUDA device visible to CuPy')
    return cupy


def _cupy_kernel() -> str:
    cp = _cupy_device()
    kernel = cp.RawKernel(_COOP_SOURCE, 'neurofly_probe', options=('-std=c++17',),
                          enable_cooperative_groups=True)
    out = cp.zeros(32, dtype=cp.int32)
    kernel((1,), (32,), (out,))
    if not bool((out == cp.arange(1, 33, dtype=cp.int32)).all()):
        raise RuntimeError('CuPy probe kernel returned wrong values')
    return 'CuPy cooperative kernel compiled and ran'


def _cupy_sparse() -> str:
    ok, reason = cupy_kernel()
    if not ok:
        raise _Absent(reason)
    import cupy as cp
    import cupyx.scipy.sparse as csp
    m = csp.csr_matrix((cp.ones(3, dtype=cp.float32), cp.array([0, 1, 2], dtype=cp.int32),
                        cp.array([0, 1, 2, 3], dtype=cp.int32)), shape=(3, 3))
    y = m @ cp.arange(3, dtype=cp.float32)
    if not bool((y == cp.arange(3, dtype=cp.float32)).all()):
        raise RuntimeError('CuPy sparse probe returned wrong values')
    return 'CuPy kernel and cuSPARSE product ran'


def numba_cuda() -> Tuple[bool, str]:
    """(usable, reason) for the numba.cuda v3 build."""
    return _run('numba.cuda', _numba_cuda)


def cupy_kernel() -> Tuple[bool, str]:
    """(usable, reason) for the CuPy v3 build (NVRTC + cooperative launch)."""
    return _run('CuPy', _cupy_kernel)


def cupy_sparse() -> Tuple[bool, str]:
    """(usable, reason) for the CuPy v4/v5 builds (adds cuSPARSE)."""
    return _run('CuPy sparse (v4/v5)', _cupy_sparse)


def describe(dynamics: str) -> Tuple[str, str]:
    """('cuda'|'cpu', reason) that an 'auto' Brain with these dynamics would pick."""
    if dynamics in ('v4', 'v5'):
        ok, reason = cupy_sparse()
        return ('cuda', reason) if ok else ('cpu', reason)
    if dynamics != 'v3':
        return 'cpu', f'LIF {dynamics} has no GPU build'
    reasons = []
    for probe in (numba_cuda, cupy_kernel):
        ok, reason = probe()
        if ok:
            return 'cuda', reason
        reasons.append(reason)
    return 'cpu', '; '.join(reasons)


def explain(dynamics: str) -> Tuple[str, str]:
    """('cuda'|'cpu', reason) honouring NEUROFLY_BRAIN_BACKEND, for logs and status."""
    import os
    forced = os.environ.get('NEUROFLY_BRAIN_BACKEND')
    if forced and forced != 'auto':
        return forced, f'forced by NEUROFLY_BRAIN_BACKEND={forced}'
    return describe(dynamics)


def reset() -> None:
    """Forget cached results (tests)."""
    _RESULTS.clear()
