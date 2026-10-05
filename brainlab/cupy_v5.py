"""GPU (CuPy) implementation of LIF v5: v4 plus per-receptor-class kinetics.

Same equations, bounds, schedule, reset and release function as
``engine.advance_v5``; see ``docs/LIF_DYNAMICS_SPEC.md`` §9.

It is ``cupy_v4.CupyV4State`` generalised from one excitatory and one
inhibitory conductance to one excitatory and one inhibitory conductance per
KINETIC CHANNEL (a set of receptor classes sharing a decay constant).  Each
(channel, sign) pair gets its own in-edge CSR matrix built from the edges whose
PRESYNAPTIC neuron belongs to that channel:

    A[k, exc][j, i] = |w_ij| * g_unit_exc * q_k   for w_ij > 0, chan(i) == k
    A[k, inh][j, i] = |w_ij| * g_unit_inh * q_k   for w_ij < 0, chan(i) == k

and each tick ``g[k,s] = g[k,s] * a_k + A[k,s] @ x``, with the same emission
vector ``x`` v4 uses (1.0 on a delivered spike, ``r(V)*dt/1000`` release events
for a graded cell).

**Equivalence with v4.**  When every receptor class has v4's 5 ms there is ONE
channel, ``q = 1.0``, and the two matrices are built exactly as
``CupyV4State.set_weights`` builds them (all edges, explicit zeros kept), so
every cupy operation below is the one v4 executes, in the same order:
bit-identical on the same GPU (``tests/test_kinetics_v5.py``).  With several
channels the matrices keep only their own nonzero edges, so the summed nnz over
all of them equals the graph's edge count -- half of what v4's two all-edge
matrices traverse.
"""
from __future__ import annotations

import numpy as np

from .cupy_v4 import _transposed_csr
from .engine import (DELAY_MS, E_EXC_MV, R_MAX_HZ, REFRACTORY_MS, TAU_M_MS,
                     V_RESET_MV, V_REST_MV, V_THRESHOLD_MV)


class CupyV5State:
    """Device-resident graph and v5 state for one ``Brain``."""

    def __init__(self, ptr, post, weight, *, n, e_inh, g_unit_exc, g_unit_inh, dt,
                 delay_slots, graded, pre_chan, chan_decay, chan_q):
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
        self.chan_decay = [float(a) for a in chan_decay]
        self.chan_q = [float(q) for q in chan_q]
        self.n_chan = len(self.chan_decay)
        self.pre_chan = np.ascontiguousarray(pre_chan, dtype=np.int64)
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
        edge_chan = np.repeat(self.pre_chan, np.diff(self._ptr))
        # (channel, sign) -> matrix, or None when that pair has no edge at all.
        self.A = []
        for k in range(self.n_chan):
            row = []
            for sign_sel, quantum in ((w > 0, self.g_unit_exc), (w < 0, self.g_unit_inh)):
                if self.n_chan == 1:
                    # Exactly CupyV4State.set_weights: all edges, zeros kept,
                    # and q = 1.0 is not even multiplied in.
                    sel = sign_sel
                    data = np.where(sel, np.abs(w) * quantum, 0.0).astype(np.float32)
                    if self.chan_q[0] != 1.0:
                        data = np.where(sel, np.abs(w) * quantum * self.chan_q[0],
                                        0.0).astype(np.float32)
                    host = _transposed_csr(self._ptr, self._post, data, self.n)
                    host.data[host.data == 0] = 0.0
                else:
                    sel = sign_sel & (edge_chan == k)
                    if not sel.any():
                        row.append(None)
                        continue
                    data = np.where(sel, np.abs(w) * quantum * self.chan_q[k],
                                    0.0).astype(np.float32)
                    host = _transposed_csr(self._ptr, self._post, data, self.n)
                    host.eliminate_zeros()
                row.append(csp.csr_matrix(
                    (cupy.asarray(host.data), cupy.asarray(host.indices),
                     cupy.asarray(host.indptr)), shape=host.shape))
                del host, data
            self.A.append(row)
        cupy.get_default_memory_pool().free_all_blocks()

    def update_edges(self, edges, values):
        raise NotImplementedError(
            'The v5 GPU backend uploads the whole weight array; assign brain.weight instead '
            'of editing edges in place (plasticity under v5 is not implemented).')

    # -- state -------------------------------------------------------------
    def upload_state(self, v, g, refractory, queue, queue_count, counts, active_flag,
                     rel_ring=None):
        cupy = self.cp
        self.d_v = cupy.asarray(v)
        self.d_g = [cupy.asarray(g[r]) for r in range(2 * self.n_chan)]
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
        for r in range(2 * self.n_chan):
            g[r] = self.cp.asnumpy(self.d_g[r])
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
        v, refr = self.d_v, self.d_refr
        g = list(self.d_g)
        graded, spiking = self.graded, self.spiking
        e_inh = self.e_inh
        K = self.n_chan
        zero = cupy.float32(0)
        for step in range(steps):
            c = cursor + step
            slot = c % self.delay_slots
            future = (c + self.delay_ticks) % self.delay_slots
            # Phase 1: integrate (as v4), on the summed channel conductances.
            refr_dec = cupy.where(spiking & (refr > 0), refr - 1, refr)
            integrating = graded | (refr_dec == 0)
            ge = g[0]
            gi = g[1]
            for k in range(1, K):
                ge = ge + g[2 * k]
                gi = gi + g[2 * k + 1]
            gtot = 1.0 + ge + gi
            vinf = (V_REST_MV + ge * E_EXC_MV + gi * e_inh + self.d_drive) / gtot
            vi = vinf + (v - vinf) * cupy.exp(-self.dt * gtot / TAU_M_MS)
            cupy.clip(vi, e_inh, E_EXC_MV, out=vi)
            v = cupy.where(integrating, vi, v)
            for k in range(K):
                a = self.chan_decay[k]
                g[2 * k] = cupy.where(integrating, g[2 * k] * a, g[2 * k])
                g[2 * k + 1] = cupy.where(integrating, g[2 * k + 1] * a, g[2 * k + 1])
            refr = refr_dec
            # Phase 1b: emission.
            spike = spiking & (refr == 0) & (v > V_THRESHOLD_MV)
            self.d_counts += spike
            emit = cupy.where(graded, (v - e_inh) * self.rel_scale,
                              spike.astype(cupy.float32))
            self.d_emit_ring[future] = emit
            # Phase 2: deliver into each channel, dropping arrivals at a
            # refractory target.
            x = self.d_emit_ring[slot]
            open_target = (refr == 0)
            for k in range(K):
                for s in range(2):
                    A = self.A[k][s]
                    if A is None:
                        continue
                    g[2 * k + s] = g[2 * k + s] + cupy.where(open_target, A.dot(x), zero)
            self.d_emit_ring[slot] = 0
            self.d_active_flag = self.d_active_flag | (emit != 0)
            # Phase 3: reset this tick's spikers.
            v = cupy.where(spike, cupy.float32(V_RESET_MV), v)
            for r in range(2 * K):
                g[r] = cupy.where(spike, zero, g[r])
            refr = cupy.where(spike, self.refractory_ticks, refr)
        self.d_v = v.astype(cupy.float32)
        self.d_g = [x_.astype(cupy.float32) for x_ in g]
        self.d_refr = refr.astype(cupy.int32)
        counts_out[...] = cupy.asnumpy(self.d_counts).astype(counts_out.dtype)
        return int(cursor) + int(steps)

    def release_rate_hz(self) -> np.ndarray:
        cupy = self.cp
        r = cupy.where(self.graded, (self.d_v - self.e_inh) * (R_MAX_HZ / (E_EXC_MV - self.e_inh)),
                       cupy.float32(0))
        return cupy.asnumpy(r)
