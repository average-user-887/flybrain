"""GPU (CuPy) implementation of LIF v6a: v5 plus a declared per-class graded transfer gain.

Same equations, bounds, schedule, reset and release function as
``engine.advance_v6a``; see ``docs/LIF_DYNAMICS_SPEC.md`` §10.

It is ``cupy_v5.CupyV5State`` with exactly one line of arithmetic changed: the
emission of a graded cell is

    x = clip((V - rel_v0) * rel_k, 0, (E_exc - E_inh) * rel_scale)

with per-neuron ``rel_v0`` (the potential at which release reaches zero) and
``rel_k`` (release per mV, in events per dt) from the declared release table
(``brainlab.graded_release``), instead of v5's ``(V - E_inh) * rel_scale``.
With the ``v5-linear`` table every neuron has ``rel_v0 = E_inh`` and
``rel_k = rel_scale``, both exactly representable as the same float32 values
v5 uses as scalars, and the clip cannot bind inside the membrane bounds, so the
result is bit-identical to v5 on the same GPU (``tests/test_release_v6a.py``).

``cupy_v5.py`` itself is not modified: v6a subclasses it.
"""
from __future__ import annotations

import numpy as np

from .cupy_v5 import CupyV5State
from .engine import E_EXC_MV, R_MAX_HZ, TAU_M_MS, V_RESET_MV, V_REST_MV, V_THRESHOLD_MV


class CupyV6aState(CupyV5State):
    """Device-resident graph and v6a state for one ``Brain``."""

    def __init__(self, ptr, post, weight, *, rel_v0, rel_k, **kw):
        super().__init__(ptr, post, weight, **kw)
        cupy = self.cp
        self.d_rel_v0 = cupy.asarray(np.asarray(rel_v0, dtype=np.float32))
        self.d_rel_k = cupy.asarray(np.asarray(rel_k, dtype=np.float32))
        # Same float32 expression v5's bound would have: (E_exc - E_inh) * rel_scale.
        self.rel_max = cupy.float32(cupy.float32(E_EXC_MV - self.e_inh) * cupy.float32(self.rel_scale))
        self.d_rel_max_hz = None

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
            # Phase 1: integrate (as v5), on the summed channel conductances.
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
            # Phase 1b: emission -- the ONE changed line: the declared class gain.
            spike = spiking & (refr == 0) & (v > V_THRESHOLD_MV)
            self.d_counts += spike
            release = cupy.clip((v - self.d_rel_v0) * self.d_rel_k, zero, self.rel_max)
            emit = cupy.where(graded, release, spike.astype(cupy.float32))
            self.d_emit_ring[future] = emit
            # Phase 2: deliver into each channel (as v5).
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
            # Phase 3: reset this tick's spikers (as v5).
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
        x = cupy.clip((self.d_v - self.d_rel_v0) * self.d_rel_k, cupy.float32(0), self.rel_max)
        r = cupy.where(self.graded, x * cupy.float32(1000.0 / self.dt), cupy.float32(0))
        return cupy.asnumpy(r)
