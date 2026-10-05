"""GPU (CuPy) implementation of the v4 hybrid graded/spiking LIF dynamics.

Same equations, bounds, schedule, reset and release function as
``engine.advance_v4``; see ``docs/LIF_DYNAMICS_SPEC.md`` §7.

Why this build exists at all.  Under v3, delivery costs work only when a
neuron spikes, and the CPU kernel walks a short queue.  Under v4 every
declared graded cell emits on *every* tick, so delivery is dense over the
graded population's out-edges -- 57 % of the graph under the primary
declaration.  The CPU reference kernel is then far too slow for the real
graph, and it stays the reference for correctness, not for measurement.

The trick that makes it affordable is to express one tick's delivery as two
sparse matrix-vector products over an *emission* vector that is uniform across
both kinds of cell:

* a spiking cell contributes ``1.0`` on the tick its spike is delivered;
* a graded cell contributes ``r(V)*dt/1000`` release events, every tick.

That is exactly what §7.3 derives: a graded cell at release rate ``r``
delivers the mean conductance a spiking cell firing at ``r`` Hz delivers.  So
one pair of matrices, built once from the pinned weights, handles the whole
brain:

    A_exc[j, i] = |w_ij| * g_unit_exc   for w_ij > 0
    A_inh[j, i] = |w_ij| * g_unit_inh   for w_ij < 0

and each tick ``ge += A_exc @ x``, ``gi += A_inh @ x``.  Both are stored as CSR
of the *transposed* graph (in-edge lists), so each target's conductance is a
per-row reduction: no atomics, fixed summation order, bit-reproducible on the
same GPU.  Arrivals at a refractory target are masked out after the product,
which is the same rule the CPU kernel applies inside the loop.

The integration pass is dense over all neurons rather than over an active set.
That changes no arithmetic: a neuron with no drive, no conductance and ``v =
V_rest`` integrates to ``V_rest``.  It is not guaranteed bit-identical to the
CPU kernel, for the same float-rounding reason ``cuda_engine`` documents for
v3; ``tests/test_graded_v4.py`` checks the two agree closely on a small graph.
"""
from __future__ import annotations

import numpy as np

from .engine import (DELAY_MS, E_EXC_MV, R_MAX_HZ, REFRACTORY_MS, TAU_M_MS,
                     TAU_SYN_MS, V_RESET_MV, V_REST_MV, V_THRESHOLD_MV)


def cupy_available() -> bool:
    """True only when CuPy kernels and cuSPARSE really run (brainlab.gpu_probe)."""
    from .gpu_probe import cupy_sparse
    return cupy_sparse()[0]


def _transposed_csr(ptr, post, data, n):
    """CSR of the transpose of the (pre -> post) matrix: in-edge lists."""
    import scipy.sparse as sp
    m = sp.csr_matrix((data, post.astype(np.int32), ptr.astype(np.int64)), shape=(n, n))
    m = m.T.tocsr()
    m.sort_indices()
    return m


class CupyV4State:
    """Device-resident graph and v4 state for one ``Brain``."""

    def __init__(self, ptr, post, weight, *, n, e_inh, g_unit_exc, g_unit_inh, dt,
                 delay_slots, graded):
        import cupy
        self.cp = cupy
        self.n = int(n)
        self.dt = float(dt)
        self.e_inh = float(e_inh)
        self.g_unit_exc = float(g_unit_exc)
        self.g_unit_inh = float(g_unit_inh)
        self.delay_slots = int(delay_slots)
        self.delay_ticks = int(round(DELAY_MS / self.dt))
        self.refractory_ticks = int(round(REFRACTORY_MS / self.dt))
        self.ag = float(np.exp(-self.dt / TAU_SYN_MS))
        self.rel_scale = float(R_MAX_HZ * self.dt * 1e-3 / (E_EXC_MV - self.e_inh))
        self.graded = cupy.asarray(np.asarray(graded, dtype=np.uint8).astype(bool))
        self.spiking = ~self.graded
        self._ptr = np.ascontiguousarray(ptr)
        self._post = np.ascontiguousarray(post)
        self.set_weights(weight)
        self.d_emit_ring = cupy.zeros((self.delay_slots, self.n), dtype=cupy.float32)
        self.d_drive = cupy.zeros(self.n, dtype=cupy.float32)
        self._last_drive = None

    # -- graph -------------------------------------------------------------
    def set_weights(self, weight):
        import cupy
        import cupyx.scipy.sparse as csp
        w = np.asarray(weight, dtype=np.float64)
        for name, sel, quantum in (('A_exc', w > 0, self.g_unit_exc),
                                   ('A_inh', w < 0, self.g_unit_inh)):
            data = np.where(sel, np.abs(w) * quantum, 0.0).astype(np.float32)
            host = _transposed_csr(self._ptr, self._post, data, self.n)
            host.data[host.data == 0] = 0.0
            setattr(self, name, csp.csr_matrix(
                (cupy.asarray(host.data), cupy.asarray(host.indices),
                 cupy.asarray(host.indptr)), shape=host.shape))
            del host, data
        cupy.get_default_memory_pool().free_all_blocks()

    def update_edges(self, edges, values):
        raise NotImplementedError(
            'The v4 GPU backend uploads the whole weight array; assign brain.weight instead '
            'of editing edges in place (plasticity under v4 is not implemented).')

    # -- state -------------------------------------------------------------
    def upload_state(self, v, g, refractory, queue, queue_count, counts, active_flag,
                     rel_ring=None):
        cupy = self.cp
        self.d_v = cupy.asarray(v)
        self.d_ge = cupy.asarray(g[0])
        self.d_gi = cupy.asarray(g[1])
        self.d_refr = cupy.asarray(refractory.astype(np.int32))
        self.d_counts = cupy.asarray(counts.astype(np.int32))
        self.d_active_flag = cupy.asarray(active_flag)
        if rel_ring is not None:
            self.d_emit_ring = cupy.asarray(rel_ring)
        else:
            self.d_emit_ring = cupy.zeros((self.delay_slots, self.n), dtype=cupy.float32)

    def download_state(self, v, g, refractory, queue, queue_count, active_flag,
                       rel_ring=None):
        v[...] = self.cp.asnumpy(self.d_v)
        g[0] = self.cp.asnumpy(self.d_ge)
        g[1] = self.cp.asnumpy(self.d_gi)
        refractory[...] = self.cp.asnumpy(self.d_refr).astype(refractory.dtype)
        queue.fill(0)
        queue_count.fill(0)
        active_flag[...] = self.cp.asnumpy(self.d_active_flag)
        if rel_ring is not None:
            rel_ring[...] = self.cp.asnumpy(self.d_emit_ring)

    # -- integration -------------------------------------------------------
    def advance(self, drive: np.ndarray, cursor: int, steps: int, counts_out: np.ndarray) -> int:
        cupy = self.cp
        if self._last_drive is None or not np.array_equal(self._last_drive, drive):
            self.d_drive = cupy.asarray(drive)
            self._last_drive = drive.copy()
        self.d_counts.fill(0)
        v, ge, gi, refr = self.d_v, self.d_ge, self.d_gi, self.d_refr
        graded, spiking = self.graded, self.spiking
        e_inh = self.e_inh
        for step in range(steps):
            c = cursor + step
            slot = c % self.delay_slots
            future = (c + self.delay_ticks) % self.delay_slots
            # Phase 1: integrate.  Spiking cells only when not refractory (the
            # counter is decremented first, so a cell integrates on the tick it
            # reaches 0); graded cells always.
            refr_dec = cupy.where(spiking & (refr > 0), refr - 1, refr)
            integrating = graded | (refr_dec == 0)
            gtot = 1.0 + ge + gi
            vinf = (V_REST_MV + ge * E_EXC_MV + gi * e_inh + self.d_drive) / gtot
            vi = vinf + (v - vinf) * cupy.exp(-self.dt * gtot / TAU_M_MS)
            cupy.clip(vi, e_inh, E_EXC_MV, out=vi)
            v = cupy.where(integrating, vi, v)
            ge = cupy.where(integrating, ge * self.ag, ge)
            gi = cupy.where(integrating, gi * self.ag, gi)
            refr = refr_dec
            # Phase 1b: emission.  Spike (1.0) or graded release events.
            spike = spiking & (refr == 0) & (v > V_THRESHOLD_MV)
            self.d_counts += spike
            emit = cupy.where(graded, (v - e_inh) * self.rel_scale,
                              spike.astype(cupy.float32))
            self.d_emit_ring[future] = emit
            # Phase 2: deliver the emission due this tick, dropping arrivals at
            # a refractory target, exactly as the CPU kernel does.
            x = self.d_emit_ring[slot]
            open_target = (refr == 0)
            ge = ge + cupy.where(open_target, self.A_exc.dot(x), cupy.float32(0))
            gi = gi + cupy.where(open_target, self.A_inh.dot(x), cupy.float32(0))
            self.d_emit_ring[slot] = 0
            self.d_active_flag = self.d_active_flag | (emit != 0)
            # Phase 3: reset this tick's spikers (after delivery, so arrivals at
            # a spiker are discarded).
            v = cupy.where(spike, cupy.float32(V_RESET_MV), v)
            ge = cupy.where(spike, cupy.float32(0), ge)
            gi = cupy.where(spike, cupy.float32(0), gi)
            refr = cupy.where(spike, self.refractory_ticks, refr)
        self.d_v, self.d_ge, self.d_gi, self.d_refr = (v.astype(cupy.float32),
                                                       ge.astype(cupy.float32),
                                                       gi.astype(cupy.float32),
                                                       refr.astype(cupy.int32))
        counts_out[...] = cupy.asnumpy(self.d_counts).astype(counts_out.dtype)
        return int(cursor) + int(steps)

    def release_rate_hz(self) -> np.ndarray:
        """Current release rate per neuron in s^-1 (0 for spiking cells)."""
        cupy = self.cp
        r = cupy.where(self.graded, (self.d_v - self.e_inh) * (R_MAX_HZ / (E_EXC_MV - self.e_inh)),
                       cupy.float32(0))
        return cupy.asnumpy(r)
