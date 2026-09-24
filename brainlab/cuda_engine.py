"""GPU (CUDA) implementation of the v3 LIF dynamics.

Same equations, bounds, schedule and reset as ``engine.advance_v3``; see
``docs/LIF_DYNAMICS_SPEC.md``.  One cooperative kernel runs a whole
``Brain.step`` (e.g. 20 ticks of 0.1 ms) with grid-wide barriers between the
phases of each tick, so there is one launch per step instead of three per tick.

Each tick has three phases, separated by grid barriers:

1. **Integrate** every active neuron (dense pass over all neurons, masked by
   ``active_flag``; a non-zero drive activates a neuron exactly as the CPU
   path does before stepping).  Spikers are appended to the delay queue slot
   18 ticks ahead and flagged for reset.
2. **Deliver** the queue slot due this tick: one warp per spiking neuron walks
   its CSR out-edges.  Arrivals at refractory neurons are dropped (as on CPU).
3. **Fold and reset**: arrivals are folded into the conductances, then every
   neuron that spiked this tick is reset.  Arrivals at a spiker are discarded,
   matching the CPU order (delivery, then reset).

Determinism: arrivals are accumulated in 64-bit integer fixed point
(``FIXED_SCALE``), so the sum does not depend on the order in which parallel
threads add them and a run is bit-reproducible on the same GPU.  The only
difference from the CPU kernel is that rounding (increments are rounded to
2**-32 leak units per edge, and per-tick sums are added to g in one float32
operation); spike trains therefore agree with the CPU reference closely but
are not guaranteed bit-identical.  ``tests/test_cuda_engine.py`` checks this.

Two builds share this design: this numba.cuda kernel (also runnable on
numba's CUDA simulator for tests) and a CuPy translation in
``cupy_engine`` for hosts where numba's CUDA target is not installed.

Only v3 is implemented.  The edge weights are uploaded once: the ``Brain``
exposes them read-only while this backend is active, so an in-place weight
edit (e.g. plasticity) fails loudly instead of being silently ignored.
"""
import math

import numpy as np
from numba import cuda

from .engine import (DELAY_MS, E_EXC_MV, REFRACTORY_MS, TAU_M_MS, TAU_SYN_MS,
                     V_RESET_MV, V_REST_MV, V_THRESHOLD_MV)

FIXED_SCALE = float(2 ** 32)
WARP = 32
THREADS_PER_BLOCK = 256


def numba_cuda_available() -> bool:
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


_AVAILABLE = None


def cuda_available() -> bool:
    """True when either GPU build (numba.cuda or CuPy) can run on this host (cached)."""
    global _AVAILABLE
    if _AVAILABLE is None:
        if numba_cuda_available():
            _AVAILABLE = True
        else:
            from .cupy_engine import cupy_available
            _AVAILABLE = cupy_available()
    return _AVAILABLE


def make_state(*args, **kwargs):
    """Device state from the first usable build: numba.cuda, else CuPy.

    ``NEUROFLY_CUDA_IMPL=numba|cupy`` forces one.  numba's CUDA simulator
    (``NUMBA_ENABLE_CUDASIM=1``) counts as numba.cuda, which is how the tests
    exercise the kernel on machines without a GPU.
    """
    import os
    impl = os.environ.get('NEUROFLY_CUDA_IMPL')
    if impl == 'numba' or (impl is None and numba_cuda_available()):
        return CudaV3State(*args, **kwargs)
    if impl in (None, 'cupy'):
        from .cupy_engine import CupyV3State, cupy_available
        if impl == 'cupy' or cupy_available():
            return CupyV3State(*args, **kwargs)
    raise RuntimeError('No CUDA build available: install CuPy or numba-cuda, or set '
                       'NUMBA_ENABLE_CUDASIM=1 for the (slow) simulator')


@cuda.jit
def _advance_v3_kernel(ptr, post, edge_inc, v, g, acc, refractory, drive, queue, queue_count,
                       counts, active_flag, spiked, cursor0, steps, dt, e_inh):
    grid = cuda.cg.this_grid()
    tid = cuda.grid(1)
    nthreads = cuda.gridsize(1)
    n = v.shape[0]
    delay_slots = queue.shape[0]
    delay_ticks = int(round(DELAY_MS / dt))
    refractory_ticks = int(round(REFRACTORY_MS / dt))
    ag = math.exp(-dt / TAU_SYN_MS)
    warp_id = tid // WARP
    lane = tid % WARP
    nwarps = nthreads // WARP
    inv_scale = 1.0 / FIXED_SCALE

    for step in range(steps):
        cursor = cursor0 + step
        slot = cursor % delay_slots
        future = (cursor + delay_ticks) % delay_slots

        # Phase 1: integrate, threshold, enqueue.
        for i in range(tid, n, nthreads):
            if active_flag[i] == 0:
                if drive[i] != 0.0:
                    active_flag[i] = 1
                else:
                    continue
            if refractory[i] > 0:
                refractory[i] -= 1
            if refractory[i] == 0:
                ge = g[0, i]
                gi = g[1, i]
                gtot = 1.0 + ge + gi
                vinf = (V_REST_MV + ge * E_EXC_MV + gi * e_inh + drive[i]) / gtot
                vi = vinf + (v[i] - vinf) * math.exp(-dt * gtot / TAU_M_MS)
                if vi < e_inh:
                    vi = e_inh
                elif vi > E_EXC_MV:
                    vi = E_EXC_MV
                v[i] = vi
                g[0, i] = ge * ag
                g[1, i] = gi * ag
                if vi > V_THRESHOLD_MV:
                    counts[i] += 1
                    spiked[i] = 1
                    k = cuda.atomic.add(queue_count, future, 1)
                    queue[future, k] = i
        grid.sync()

        # Phase 2: deliver the spikes due this tick, one warp per spiker.
        nspk = queue_count[slot]
        for q in range(warp_id, nspk, nwarps):
            pre = queue[slot, q]
            for e in range(ptr[pre] + lane, ptr[pre + 1], WARP):
                j = post[e]
                if refractory[j] > 0:
                    continue
                inc = edge_inc[e]
                if inc > 0:
                    cuda.atomic.add(acc, (0, j), inc)
                else:
                    cuda.atomic.add(acc, (1, j), -inc)
                active_flag[j] = 1
        grid.sync()

        # Phase 3: fold arrivals into g, reset this tick's spikers.
        if tid == 0:
            queue_count[slot] = 0
        for i in range(tid, n, nthreads):
            if spiked[i] != 0:
                spiked[i] = 0
                v[i] = V_RESET_MV
                g[0, i] = 0.0
                g[1, i] = 0.0
                refractory[i] = refractory_ticks
                acc[0, i] = 0
                acc[1, i] = 0
            else:
                a0 = acc[0, i]
                a1 = acc[1, i]
                if a0 != 0:
                    g[0, i] += a0 * inv_scale
                    acc[0, i] = 0
                if a1 != 0:
                    g[1, i] += a1 * inv_scale
                    acc[1, i] = 0
        grid.sync()


@cuda.jit
def _scatter_int64(dst, idx, values):
    k = cuda.grid(1)
    if k < idx.shape[0]:
        dst[idx[k]] = values[k]


# Device copies of immutable graph arrays, shared by every Brain on this
# process (e.g. the daemon's per-assay instances).  Keyed by the host buffer;
# only read-only host arrays are cached, so a key can never go stale.
_DEVICE_CACHE = {}


def cached_device_array(impl: str, host: np.ndarray, upload, build=None, tag=''):
    """Upload ``build(host)``; reuse one device copy per read-only host array."""
    build = build or (lambda a: a)
    if host.flags.writeable:
        return upload(build(host))
    key = (impl, host.__array_interface__['data'][0], host.nbytes, host.dtype.str, tag)
    if key not in _DEVICE_CACHE:
        # Holding ``host`` keeps its buffer alive, so its address is never reused.
        _DEVICE_CACHE[key] = (host, upload(build(host)))
    return _DEVICE_CACHE[key][1]


def edge_increments(weight: np.ndarray, g_unit_exc: float, g_unit_inh: float) -> np.ndarray:
    """Per-edge conductance increment in fixed point; the sign marks inhibition."""
    w = weight.astype(np.float64)
    inc = np.where(w > 0, w * g_unit_exc, w * g_unit_inh)   # inhibitory stays negative
    return np.ascontiguousarray(np.rint(inc * FIXED_SCALE).astype(np.int64))


class CudaV3State:
    """Device-resident graph and v3 state for one ``Brain``."""

    def __init__(self, ptr, post, weight, *, n, e_inh, g_unit_exc, g_unit_inh, dt,
                 delay_slots, blocks=None):
        self.n = int(n)
        self.dt = float(dt)
        self.e_inh = float(e_inh)
        self.g_unit_exc = float(g_unit_exc)
        self.g_unit_inh = float(g_unit_inh)
        self.d_ptr = cached_device_array('numba', ptr, cuda.to_device)
        self.d_post = cached_device_array('numba', post, cuda.to_device)
        self.set_weights(weight)
        self.d_acc = cuda.to_device(np.zeros((2, self.n), dtype=np.int64))
        self.d_spiked = cuda.to_device(np.zeros(self.n, dtype=np.uint8))
        self.d_drive = cuda.device_array(self.n, dtype=np.float32)
        self.delay_slots = int(delay_slots)
        self._blocks = blocks
        self._last_drive = None

    def _increments(self, weight):
        return edge_increments(weight, self.g_unit_exc, self.g_unit_inh)

    def set_weights(self, weight):
        """(Re)upload all edge weights; shared read-only weights share one device copy."""
        self.d_edge_inc = cached_device_array('numba', weight, cuda.to_device, self._increments,
                                              tag=f'inc {self.g_unit_exc!r} {self.g_unit_inh!r}')
        self._edge_inc_private = weight.flags.writeable

    def update_edges(self, edges, values):
        """Rewrite the increments of ``edges`` (e.g. plastic synapses) on the device."""
        if not self._edge_inc_private:   # never write into a shared device copy
            self.d_edge_inc = cuda.to_device(self.d_edge_inc.copy_to_host())
            self._edge_inc_private = True
        if len(edges):
            inc = self._increments(np.asarray(values, dtype=np.float32))
            _scatter_int64[(len(edges) + 255) // 256, 256](
                self.d_edge_inc, cuda.to_device(np.ascontiguousarray(edges, dtype=np.int64)),
                cuda.to_device(inc))

    def upload_state(self, v, g, refractory, queue, queue_count, counts, active_flag):
        self.d_v = cuda.to_device(v)
        self.d_g = cuda.to_device(g)
        self.d_refractory = cuda.to_device(refractory)
        self.d_queue = cuda.to_device(queue)
        self.d_queue_count = cuda.to_device(queue_count)
        self.d_counts = cuda.to_device(counts)
        self.d_active_flag = cuda.to_device(active_flag)

    def download_state(self, v, g, refractory, queue, queue_count, active_flag):
        self.d_v.copy_to_host(v)
        self.d_g.copy_to_host(g)
        self.d_refractory.copy_to_host(refractory)
        self.d_queue.copy_to_host(queue)
        self.d_queue_count.copy_to_host(queue_count)
        self.d_active_flag.copy_to_host(active_flag)

    def _grid_blocks(self):
        if self._blocks is None:
            blocks = None
            try:
                sig = next(iter(_advance_v3_kernel.overloads))
                blocks = _advance_v3_kernel.overloads[sig].max_cooperative_grid_blocks(THREADS_PER_BLOCK)
            except Exception:
                blocks = None
            if not blocks:
                try:
                    blocks = int(cuda.get_current_device().MULTIPROCESSOR_COUNT)
                except Exception:   # numba's CUDA simulator has no device query
                    blocks = 1
            needed = (self.n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            self._blocks = int(max(1, min(blocks, needed)))
        return self._blocks

    def advance(self, drive: np.ndarray, cursor: int, steps: int, counts_out: np.ndarray) -> int:
        if self._last_drive is None or not np.array_equal(self._last_drive, drive):
            self.d_drive.copy_to_device(drive)
            self._last_drive = drive.copy()
        self.d_counts.copy_to_device(np.zeros(self.n, dtype=np.int32))
        args = (self.d_ptr, self.d_post, self.d_edge_inc, self.d_v, self.d_g, self.d_acc,
                self.d_refractory, self.d_drive, self.d_queue, self.d_queue_count,
                self.d_counts, self.d_active_flag, self.d_spiked, int(cursor), int(steps),
                self.dt, self.e_inh)
        if self._blocks is None and not getattr(_advance_v3_kernel, 'overloads', True):
            # Compile at the real launch shape once so the occupancy query works.
            _advance_v3_kernel[1, THREADS_PER_BLOCK](*args[:13], int(cursor), 0, self.dt, self.e_inh)
        _advance_v3_kernel[self._grid_blocks(), THREADS_PER_BLOCK](*args)
        self.d_counts.copy_to_host(counts_out)
        return int(cursor) + int(steps)
