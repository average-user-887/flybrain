"""v8 input timing: training targets, protocols, measurements, objective and gates (pure logic plus
a small protocol runner over a BrainV6).  Targets are Drosophila NON-MOTION intracellular voltage
timing of R1-R6 and LMCs (qualification/v8/V8_data_pin.json).  The held-out medulla data
are NOT used here (heldout_v8.py).
"""
from __future__ import annotations

import copy
import math

import numpy as np

# --- targets (V8_data_pin.json) -------------------------------------------------------------
# Mansour et al. 2026 Supp. Fig. 1b: 10-ms bright flash after brief dark adaptation, 25 C.
T_FLASH_R_TP = (25.5, 2.4)          # mean, dispersion (n=4)
T_FLASH_LMC_TP = (12.2, 1.1)        # (n=5), LMC subtype pooled -> applied to L1 and L2
T_FLASH_LATENCY = (6.5, 1.5)        # '~6-7 ms', both cell types; sigma = declared reading tolerance
# Juusola & Hardie 2001: impulse time to peak and dead time at the background endpoints.
T_IMPULSE_TP = {'BG-4': 40.0, 'BG0': 20.0}
T_DEAD = {'BG-4': 20.0, 'BG0': 10.0}
SIG_TP_FRAC, SIG_DEAD_FRAC = 0.20, 0.25

# --- protocols ------------------------------------------------------------------------------
DARK_ADAPT_MS = 10_000              # Mansour: 5-15 s dark adaptation
FLASH_MS, FLASH_LEVEL = 10, 330.0   # encoder 330 = 3.3 x BG0 (JH2001's brightest impulse); ASSUMPTION
FLASH_RECORD_MS = 100
BACKGROUND = {'BG-4': 0.01, 'BG0': 100.0}   # encoder level (x 0.01 = BG0 units)
BG_SETTLE_MS = 3000
IMPULSE_MS, IMPULSE_FACTOR = 1, 3.0  # 1-ms pulse to 3 x background (contrast +2); ASSUMPTION
IMPULSE_RECORD_MS = 150
ONSET_FRAC = 0.10                   # onset/latency = first |dV| >= 10 % of the peak deflection

LMC_TYPES = ('L1', 'L2')


def onset_tp(x, frac=ONSET_FRAC):
    """(onset_ms, time_to_peak_ms, peak) of a deflection trace sampled at 1 ms from stimulus onset."""
    x = np.asarray(x, float)
    k = int(np.argmax(np.abs(x)))
    above = np.flatnonzero(np.abs(x[:k + 1]) >= frac * abs(x[k]))
    return (float(above[0]) if len(above) else math.nan), float(k), float(x[k])


def apply_input_params(b, idx, light, pt_base, fitted):
    """Write the v8 free parameters into a BrainV6 (in memory).  Phototransduction keys are global
    to R1-R6; L1/L2 keys per type.  Synapses (release) and the medulla are never touched."""
    from brainlab.v6_calibrated import Phototransduction
    pt = copy.deepcopy(pt_base)
    for k, v in fitted.get('R1-R6', {}).items():
        if k not in ('tau_p0_ms', 'Kt', 'dead_time_ms'):
            raise ValueError(f'R1-R6 key {k} is not a v8 free parameter')
        pt[k] = int(round(v)) if k == 'dead_time_ms' else float(v)
    b.v6_params = dict(b.v6_params, phototransduction=pt)
    b.pt = Phototransduction(pt, len(light))
    for t in LMC_TYPES:
        i = idx[t]
        for k, v in fitted.get(t, {}).items():
            if k == 'tau_m_ms':
                b.tau_m[i] = v
            elif k == 'ih_g':
                b.gh[i] = v
            elif k == 'ih_tau_ms':
                b.h_tau[i] = v
            elif k == 'ih_vh_mV':
                b.h_vh[i] = v
            else:
                raise ValueError(f'{t} key {k} is not a v8 free parameter')
    for t in fitted:
        if t not in ('R1-R6',) + LMC_TYPES:
            raise ValueError(f'type {t} is not free in v8 (medulla and synapses are fixed)')


def run_levels(b, light, cols, levels):
    drive = np.zeros(b.n, np.float32)
    out = np.zeros((len(levels), len(cols)))
    for k, lv in enumerate(levels):
        drive[light] = lv
        b.step(drive, 1.0)
        out[k] = b.v[cols]
    return out


def measure(b, light, cols):
    """cols: dict name -> node index for 'R', 'L1', 'L2'.  Returns all protocol measurements."""
    names = list(cols)
    c = [cols[n] for n in names]
    res = {}
    b.reset_state()
    pre = run_levels(b, light, c, np.zeros(DARK_ADAPT_MS))
    V = run_levels(b, light, c, np.r_[np.full(FLASH_MS, FLASH_LEVEL), np.zeros(FLASH_RECORD_MS - FLASH_MS)])
    res['flash'] = {n: dict(zip(('onset_ms', 'tp_ms', 'peak_mV'), onset_tp(V[:, j] - pre[-1, j])))
                    for j, n in enumerate(names)}
    for bg, lv in BACKGROUND.items():
        b.reset_state()
        pre = run_levels(b, light, c, np.full(BG_SETTLE_MS, lv))
        V = run_levels(b, light, c, np.r_[np.full(IMPULSE_MS, lv * IMPULSE_FACTOR),
                                           np.full(IMPULSE_RECORD_MS - IMPULSE_MS, lv)])
        res[bg] = {n: dict(zip(('onset_ms', 'tp_ms', 'peak_mV'), onset_tp(V[:, j] - pre[-1, j])))
                   for j, n in enumerate(names)}
    return res


def _sq(x, mu, sd):
    return ((x - mu) / sd) ** 2 if np.isfinite(x) else 1e6


def objective(m):
    """Sum of squared standardised errors over the training targets (finite-safe)."""
    c = _sq(m['flash']['R']['tp_ms'], *T_FLASH_R_TP) + _sq(m['flash']['R']['onset_ms'], *T_FLASH_LATENCY)
    for t in LMC_TYPES:
        c += _sq(m['flash'][t]['tp_ms'], *T_FLASH_LMC_TP) + _sq(m['flash'][t]['onset_ms'], *T_FLASH_LATENCY)
    for bg in BACKGROUND:
        c += _sq(m[bg]['R']['tp_ms'], T_IMPULSE_TP[bg], SIG_TP_FRAC * T_IMPULSE_TP[bg])
        c += _sq(m[bg]['R']['onset_ms'], T_DEAD[bg], SIG_DEAD_FRAC * T_DEAD[bg])
    return float(c)


# --- training gates (frozen in V8_input_prereg.json) -------------------------------------------
G_FLASH_SD = 2.0
G_LATENCY = (4.0, 10.0)
G_LATENCY_DIFF = 3.0
G_TP_FRAC, G_DEAD_FRAC = 0.25, 0.30


def train_gates(m):
    f = m['flash']
    rlo, rhi = T_FLASH_R_TP[0] - G_FLASH_SD * T_FLASH_R_TP[1], T_FLASH_R_TP[0] + G_FLASH_SD * T_FLASH_R_TP[1]
    llo, lhi = T_FLASH_LMC_TP[0] - G_FLASH_SD * T_FLASH_LMC_TP[1], T_FLASH_LMC_TP[0] + G_FLASH_SD * T_FLASH_LMC_TP[1]
    g1 = rlo <= f['R']['tp_ms'] <= rhi
    g2 = {t: (llo <= f[t]['tp_ms'] <= lhi) and f[t]['tp_ms'] < f['R']['tp_ms'] for t in LMC_TYPES}
    g3 = {n: G_LATENCY[0] <= f[n]['onset_ms'] <= G_LATENCY[1] for n in ('R',) + LMC_TYPES}
    g3['no_synaptic_delay'] = all(abs(f[t]['onset_ms'] - f['R']['onset_ms']) <= G_LATENCY_DIFF for t in LMC_TYPES)
    g4 = {bg: abs(m[bg]['R']['tp_ms'] - T_IMPULSE_TP[bg]) <= G_TP_FRAC * T_IMPULSE_TP[bg] for bg in BACKGROUND}
    g4['adapts'] = m['BG0']['R']['tp_ms'] < m['BG-4']['R']['tp_ms']
    g5 = {bg: abs(m[bg]['R']['onset_ms'] - T_DEAD[bg]) <= G_DEAD_FRAC * T_DEAD[bg] for bg in BACKGROUND}
    ok = g1 and all(g2.values()) and all(g3.values()) and all(g4.values()) and all(g5.values())
    return dict(G1_flash_R_tp=g1, G2_flash_LMC_tp=g2, G3_latency=g3, G4_impulse_tp=g4, G5_dead_time=g5,
                TRAIN_PASS=bool(ok))


MIN_COMPLETED_FRACTION = 0.5


def outcome(reports, n_runs):
    """As v7 v2: fewer than half of the runs completed -> NOT_EVALUABLE (never F1); else the
    lowest-cost run among those whose OWN gates pass (tie -> lowest run id); none -> F1."""
    done = [r for r in reports if r.get('status') == 'completed']
    if len(done) < MIN_COMPLETED_FRACTION * n_runs:
        return 'NOT_EVALUABLE', None
    passing = [r for r in done if r['gates']['TRAIN_PASS']]
    if not passing:
        return 'F1', None
    return 'TRAIN_PASS', min(passing, key=lambda r: (r['cost'], r['run']))
