"""LIF dynamics v6: v4's hybrid graded/spiking engine plus FROZEN per-type
parameters fitted to published NON-MOTION recordings (owner ruling, 7 Oct 2026).

This module is additive.  It does not modify v3, v4 or v5, the graph, any
edge or any base synaptic weight.  v6 differs from v4 only in quantities that
belong to a CELL TYPE:

* ``R1-R6`` receive light through a graded phototransduction model (a
  light-gated conductance reversing at 0 mV) instead of an injected current;
* a graded presynaptic type may release through a fitted sigmoid
  ``x = gain * R_MAX * dt * 1/(1 + exp(-(V - vh)/s))`` instead of v4's linear
  ``r(V)``; a type without a sigmoid keeps v4's linear release times ``gain``
  (gain 1 = v4 exactly).  ``gain`` is an EFFECTIVE release gain recorded per
  presynaptic type; it never rewrites a weight;
* a graded type may have its own membrane time constant;
* a graded type may carry an H-current (S2 hypothesis, L1/L2 only).

Every parameter comes from a frozen JSON file whose sha256 the caller passes;
:class:`BrainV6` refuses any other file.  With an empty parameter set
(``{"types": {}}`` and no phototransduction) ``advance_v6`` reduces to
``advance_v4`` arithmetic (tested in ``tests/test_v6_calibrated.py``).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .brain import Brain
from .engine import (E_EXC_MV, KERNEL_OPTIONS, R_MAX_HZ, REFRACTORY_MS, DELAY_MS, TAU_M_MS,
                     TAU_SYN_MS, V_REST_MV, V_RESET_MV, V_THRESHOLD_MV, njit)

E_LIGHT_MV = E_EXC_MV          # light-gated TRP/TRPL conductance; capped by the engine ceiling


@njit(**KERNEL_OPTIONS)
def advance_v6(ptr, post, weight, v, g, refractory, drive, queue, queue_count, cursor, steps, dt,
               counts, active, active_flag, nactive, e_inh, g_unit_exc, g_unit_inh, graded,
               graded_idx, rel_ring, tau_m, g_light, rel_gain, rel_vh, rel_s,
               gh, h_vh, h_k, h_tau, h_e, h_state):
    """``advance_v4`` with per-neuron graded-cell parameters (see module doc).

    Spiking cells are integrated exactly as in v4 (global TAU_M_MS).  For a
    graded cell: membrane time constant ``tau_m[i]``, an extra light
    conductance ``g_light[i]`` reversing at E_LIGHT_MV, an optional H-current
    ``gh[i] * h`` reversing at ``h_e[i]`` whose gate relaxes to
    ``1/(1+exp((V-h_vh)/h_k))`` with ``h_tau[i]``, and release
    ``rel_gain * sigmoid`` when ``rel_s[i] > 0`` else ``rel_gain * v4-linear``.
    """
    ag = math.exp(-dt / TAU_SYN_MS)
    delay_slots = queue.shape[0]
    delay_ticks = int(round(DELAY_MS / dt))
    refractory_ticks = int(round(REFRACTORY_MS / dt))
    n_graded = graded_idx.shape[0]
    rel_scale = R_MAX_HZ * dt * 1e-3 / (E_EXC_MV - e_inh)
    sig_scale = R_MAX_HZ * dt * 1e-3
    for step in range(steps):
        slot = cursor % delay_slots
        future = (cursor + delay_ticks) % delay_slots
        for k in range(nactive[0]):
            i = active[k]
            if graded[i] != 0:
                continue
            if refractory[i] > 0:
                refractory[i] -= 1
            if refractory[i] == 0:
                ge = g[0, i]; gi = g[1, i]
                gtot = 1.0 + ge + gi
                vinf = (V_REST_MV + ge * E_EXC_MV + gi * e_inh + drive[i]) / gtot
                vi = vinf + (v[i] - vinf) * math.exp(-dt * gtot / TAU_M_MS)
                if vi < e_inh:
                    vi = e_inh
                elif vi > E_EXC_MV:
                    vi = E_EXC_MV
                v[i] = vi
                g[0, i] = ge * ag; g[1, i] = gi * ag
                if vi > V_THRESHOLD_MV:
                    counts[i] += 1
                    queue[future, queue_count[future]] = i
                    queue_count[future] += 1
        for k in range(n_graded):
            i = graded_idx[k]
            ge = g[0, i]; gi = g[1, i]; gl = g_light[i]
            gtot = 1.0 + ge + gi + gl
            num = V_REST_MV + ge * E_EXC_MV + gi * e_inh + gl * E_LIGHT_MV + drive[i]
            if gh[i] > 0.0:
                hinf = 1.0 / (1.0 + math.exp((v[i] - h_vh[i]) / h_k[i]))
                h_state[i] = hinf + (h_state[i] - hinf) * math.exp(-dt / h_tau[i])
                gH = gh[i] * h_state[i]
                gtot += gH
                num += gH * h_e[i]
            vinf = num / gtot
            vi = vinf + (v[i] - vinf) * math.exp(-dt * gtot / tau_m[i])
            if vi < e_inh:
                vi = e_inh
            elif vi > E_EXC_MV:
                vi = E_EXC_MV
            v[i] = vi
            g[0, i] = ge * ag; g[1, i] = gi * ag
            if rel_s[i] > 0.0:
                rel_ring[future, i] = rel_gain[i] * sig_scale / (1.0 + math.exp(-(vi - rel_vh[i]) / rel_s[i]))
            else:
                rel_ring[future, i] = rel_gain[i] * (vi - e_inh) * rel_scale
        for q in range(queue_count[slot]):
            i = queue[slot, q]
            for e in range(ptr[i], ptr[i + 1]):
                j = post[e]
                if refractory[j] > 0:
                    continue
                w = weight[e]
                if w > 0.0:
                    g[0, j] += w * g_unit_exc
                else:
                    g[1, j] -= w * g_unit_inh
                if active_flag[j] == 0:
                    active_flag[j] = 1; active[nactive[0]] = j; nactive[0] += 1
        queue_count[slot] = 0
        for k in range(n_graded):
            i = graded_idx[k]
            x = rel_ring[slot, i]
            if x == 0.0:
                continue
            rel_ring[slot, i] = 0.0
            for e in range(ptr[i], ptr[i + 1]):
                j = post[e]
                if refractory[j] > 0:
                    continue
                w = weight[e]
                if w > 0.0:
                    g[0, j] += w * g_unit_exc * x
                else:
                    g[1, j] -= w * g_unit_inh * x
                if active_flag[j] == 0:
                    active_flag[j] = 1; active[nactive[0]] = j; nactive[0] += 1
        for q in range(queue_count[future]):
            i = queue[future, q]; v[i] = V_RESET_MV; g[0, i] = 0.0; g[1, i] = 0.0; refractory[i] = refractory_ticks
        cursor += 1
    return cursor


class Phototransduction:
    """Graded R1-R6 phototransduction, vectorised over photoreceptors, 1 ms steps.

    Light ``I`` (units of the JH2001 BG0 intensity, ~3e6 photons/s) passes a
    pure dead time ``D``, then ``n`` first-order stages whose time constant
    shortens with adaptation, ``tau = tau_p0 / (1 + a/Kt)`` (``Kt`` defaults to ``Ka``); ``a`` low-passes
    the cascade output with ``tau_a``.  The light conductance (leak units) is
    ``G * y / (1 + a/Ka)``: divisive gain control.  State is plain arrays so a
    snapshot captures it.
    """

    def __init__(self, p: dict, n_cells: int):
        self.G = float(p['G']); self.Ka = float(p['Ka']); self.Kt = float(p.get('Kt', p['Ka']))
        self.tau_a = float(p['tau_a_ms']); self.tau_p0 = float(p['tau_p0_ms'])
        self.n = int(p['n_stages']); self.D = int(p['dead_time_ms'])
        self.stages = np.zeros((self.n, n_cells))
        self.a = np.zeros(n_cells)
        self.ring = np.zeros((max(self.D, 1), n_cells))
        self.cur = np.zeros(1, np.int64)

    def state(self):
        return dict(pt_stages=self.stages.copy(), pt_a=self.a.copy(), pt_ring=self.ring.copy(), pt_cur=self.cur.copy())

    def set_state(self, s):
        self.stages[...] = s['pt_stages']; self.a[...] = s['pt_a']; self.ring[...] = s['pt_ring']; self.cur[...] = s['pt_cur']

    def step(self, intensity, dt_ms=1.0):
        if self.D > 0:
            k = int(self.cur[0]) % self.D
            x = self.ring[k].copy()
            self.ring[k] = intensity
            self.cur[0] += 1
        else:
            x = np.asarray(intensity, float)
        tau = self.tau_p0 / (1.0 + self.a / self.Kt)
        al = 1.0 - np.exp(-dt_ms / tau)
        prev = x
        for s in range(self.n):
            self.stages[s] += al * (prev - self.stages[s])
            prev = self.stages[s]
        y = self.stages[-1]
        self.a += (1.0 - math.exp(-dt_ms / self.tau_a)) * (y - self.a)
        return self.G * y / (1.0 + self.a / self.Ka)


def load_params(path, sha256):
    raw = Path(path).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != sha256:
        raise SystemExit(f'v6 parameter file sha256 {got} != {sha256}; refusing')
    return json.loads(raw), got


class BrainV6(Brain):
    """A v4 brain (CPU) stepped by ``advance_v6`` with frozen per-type parameters.

    ``drive`` keeps v4's meaning for every cell EXCEPT the photoreceptors named
    in ``light_nodes``: there the value is an encoder level, converted to
    light by ``phototransduction.encoder_to_intensity`` and fed to
    :class:`Phototransduction` (the injected current on those cells is zero).
    """

    def __init__(self, path, params: dict, params_sha256: str, light_nodes=None, cell_type=None, **kw):
        super().__init__(path, dynamics='v4', backend='cpu', **kw)
        if cell_type is None:
            raise ValueError('BrainV6 needs the per-node cell_type array to resolve per-type parameters')
        n = self.n
        self.v6_params = params
        self.v6_params_sha256 = params_sha256
        self.tau_m = np.full(n, TAU_M_MS, np.float64)
        self.g_light = np.zeros(n, np.float64)
        self.rel_gain = np.ones(n, np.float64)
        self.rel_vh = np.zeros(n, np.float64)
        self.rel_s = np.zeros(n, np.float64)
        self.gh = np.zeros(n, np.float64)
        self.h_vh = np.full(n, -60.0); self.h_k = np.full(n, 5.0)
        self.h_tau = np.full(n, 100.0); self.h_e = np.full(n, -30.0)
        self.h_state = np.zeros(n, np.float64)
        applied = {}
        ct = np.asarray(cell_type)
        for t, p in params.get('types', {}).items():
            m = (ct == t) & (self.graded != 0)
            idx = np.flatnonzero(m)
            applied[t] = int(len(idx))
            if 'tau_m_ms' in p:
                self.tau_m[idx] = p['tau_m_ms']
            rel = p.get('release')
            if rel:
                self.rel_gain[idx] = rel.get('gain', 1.0)
                if rel.get('kind') == 'sigmoid':
                    self.rel_vh[idx] = rel['vh_mV']; self.rel_s[idx] = rel['s_mV']
            ih = p.get('ih')
            if ih:
                self.gh[idx] = ih['g']; self.h_vh[idx] = ih['vh_mV']; self.h_k[idx] = ih['k_mV']
                self.h_tau[idx] = ih['tau_ms']; self.h_e[idx] = ih['E_mV']
        self.v6_applied = applied
        pt = params.get('phototransduction')
        self.light_nodes = np.asarray(light_nodes if light_nodes is not None else [], np.int64)
        self.pt = Phototransduction(pt, len(self.light_nodes)) if pt else None
        self.enc_to_I = float(pt['encoder_to_intensity']) if pt else 0.0
        self.dynamics_label = 'v6'

    def _state_arrays(self):
        return super()._state_arrays() + ('h_state', 'g_light')

    def snapshot_state(self):
        s = super().snapshot_state()
        if self.pt is not None:
            s.update(self.pt.state())
        s['v6_params_sha256'] = self.v6_params_sha256
        return s

    def restore_state(self, state):
        if state.get('v6_params_sha256') not in (None, self.v6_params_sha256):
            raise ValueError('snapshot written under other v6 parameters; refused')
        super().restore_state(state)
        if self.pt is not None:
            self.pt.set_state(state)

    def reset_state(self, v_rest_mV=-52.0):
        super().reset_state(v_rest_mV)
        self.h_state.fill(0.0); self.g_light.fill(0.0)
        if self.pt is not None:
            self.pt.set_state(Phototransduction(self.v6_params['phototransduction'], len(self.light_nodes)).state())

    def step(self, currents, duration_ms):
        drive = np.array(currents, dtype=np.float32)
        if drive.shape != (self.n,) or not np.isfinite(drive).all():
            raise ValueError('Provide a finite current for every neuron')
        ms = int(round(duration_ms))
        if ms < 1 or not math.isclose(ms, duration_ms):
            raise ValueError('v6 steps in whole milliseconds')
        if self.pt is not None:
            intensity = drive[self.light_nodes].astype(np.float64) * self.enc_to_I
            drive[self.light_nodes] = 0.0
        newly_active = np.flatnonzero((drive != 0) & (self.active_flag == 0))
        start = int(self.nactive[0])
        self.active[start:start + len(newly_active)] = newly_active
        self.active_flag[newly_active] = 1
        self.nactive[0] += len(newly_active)
        self.counts.fill(0)
        steps = int(round(1.0 / self.dt))
        for _ in range(ms):
            if self.pt is not None:
                self.g_light[self.light_nodes] = self.pt.step(intensity, 1.0)
            self.cursor = advance_v6(self.ptr, self.post, self.weight, self.v, self.g, self.refractory, drive,
                                     self.queue, self.queue_count, self.cursor, steps, self.dt, self.counts,
                                     self.active, self.active_flag, self.nactive, self.e_inh_mV,
                                     self.g_unit_exc, self.g_unit_inh, self.graded, self.graded_idx,
                                     self.rel_ring, self.tau_m, self.g_light, self.rel_gain, self.rel_vh,
                                     self.rel_s, self.gh, self.h_vh, self.h_k, self.h_tau, self.h_e,
                                     self.h_state)
        self.total_spikes += int(self.counts.sum())
        self.sim_ms += ms
        return self.counts.copy(), 0.0
