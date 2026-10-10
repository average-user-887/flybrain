"""v9 engine probe (prereg input, NOT a fit): CPU-only, tiny BrainV6 (one R1-R6 node -> one L1 and
one L2 node), v8 protocols and measurements (scripts/v8/v8_input.py) unchanged.

A  saturation curve: R1-R6 flash peak deflection / onset / time to peak over the half-log encoder
   grid, at the v8 base photoreceptor (S1-A3) with D = 7 and at the four corners of the v8 free
   box (tau_p0 0.5/20 ms x Kt 0.001/100).  Applies the PRE-STATED E_sat rule of
   qualification/v9/V9_flash_pin.json (committed before this script was run).
B  onset bounds at the engine's dt (0.1 ms substeps) and graded-release delay (DELAY_MS): fastest
   free values (tau_p0 0.5 ms, L tau_m 1 ms, no Ih), D = 5..11, R->L weights -50, -9.6, -5, flash at
   E_sat and at the grid top.
C  R-stage feasibility over a coarse grid of the v8 free photoreceptor box (tau_p0 x Kt, D = 7 and 8):
   noise-kernel time to peak at BG-4 and BG0 (G4) and flash time to peak at E_sat (G1).  Descriptive
   bounds only; nothing here is selected for use as fitted parameters.

Usage:  CUDA_VISIBLE_DEVICES= python scripts/v9/probe_v9.py OUT.json
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'v8'))

import v8_input as vi  # noqa: E402
from brainlab.engine import DELAY_MS  # noqa: E402

BASE_FILE = 'qualification/v6/params_v6_S2A_leak40.json'
BASE_SHA = '135220ca7e9bc01c5d88f4fee321dea7e2c5fe57c0c1db24de76187f5ee4962f'
GRID = [10 ** (k / 2) for k in range(9)]           # V9_flash_pin.json conversion_rule_prestated
SAT_FRAC = 0.90
WEIGHTS = (-50.0, -9.6, -5.0)
D_ONSET = range(5, 12)
TAU_GRID = (0.5, 1.0, 2.0, 4.327696304840035, 8.0, 20.0)
KT_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
D_FEAS = (7, 8)


def load_base():
    raw = open(os.path.join(ROOT, BASE_FILE), 'rb').read()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA, 'base params sha mismatch'
    return json.loads(raw)


def tiny(base, D, tau_p0=None, Kt=None, weight=-50.0, l_tau=1.0):
    from brainlab.v6_calibrated import BrainV6
    pt = dict(base['phototransduction'], dead_time_ms=int(D))
    if tau_p0 is not None:
        pt['tau_p0_ms'] = float(tau_p0)
    if Kt is not None:
        pt['Kt'] = float(Kt)
    types = {'R1-R6': base['types']['R1-R6'], 'L1': {'tau_m_ms': l_tau, 'e_leak_mV': -40.0},
             'L2': {'tau_m_ms': l_tau, 'e_leak_mV': -40.0}}
    arrays = dict(ptr=np.array([0, 2, 2, 2], np.int64), post=np.array([1, 2], np.int32),
                  weight=np.array([weight, weight], np.float32), ids=np.arange(1, 4, dtype=np.int64))
    b = BrainV6(None, dict(phototransduction=pt, types=types), 'e' * 64, light_nodes=np.array([0]),
                cell_type=np.array(['R1-R6', 'L1', 'L2']), arrays=arrays, graded_policy=np.ones(3, np.uint8))
    return b


def flash(b, level):
    b.reset_state()
    light = np.array([0])
    pre = vi.run_levels(b, light, [0, 1, 2], np.zeros(vi.DARK_ADAPT_MS))
    X = vi.run_levels(b, light, [0, 1, 2], np.r_[np.full(vi.FLASH_MS, level), np.zeros(vi.FLASH_RECORD_MS - vi.FLASH_MS)])
    return {n: dict(zip(('onset_ms', 'tp_ms', 'peak_mV'), vi.onset_tp(X[:, j] - pre[-1, j])))
            for j, n in enumerate(('R', 'L1', 'L2'))}


def sat_curve(base, D, tau_p0=None, Kt=None):
    rows = [dict(encoder=lv, **flash(tiny(base, D, tau_p0, Kt), lv)['R']) for lv in GRID]
    top = abs(rows[-1]['peak_mV'])
    e_sat = next((r['encoder'] for r in rows if abs(r['peak_mV']) >= SAT_FRAC * top), None)
    tps = [r['tp_ms'] for r in rows]
    mono = bool(all(tps[i + 1] <= tps[i] for i in range(len(tps) - 1)))
    return dict(rows=rows, E_sat=e_sat, tp_nonincreasing_with_level=mono)


def main(out):
    t0 = time.time()
    base = load_base()
    pt0 = base['phototransduction']
    res = dict(probe='v9 engine probe (scripts/v9/probe_v9.py)', DELAY_MS=DELAY_MS,
               substep_ms=None, encoder_to_intensity=pt0['encoder_to_intensity'], grid=GRID, sat_frac=SAT_FRAC)
    res['substep_ms'] = float(tiny(base, 7).dt)

    # A: saturation curves and E_sat
    A = {'base_D7': sat_curve(base, 7)}
    for tau in (0.5, 20.0):
        for kt in (0.001, 100.0):
            A[f'corner_tau{tau}_Kt{kt}_D7'] = sat_curve(base, 7, tau, kt)
    res['A_saturation'] = A
    E = A['base_D7']['E_sat']
    res['E_sat_fixed'] = E

    # B: onset bounds
    B = {}
    for lv in (E, GRID[-1]):
        for w in WEIGHTS:
            for D in D_ONSET:
                f = flash(tiny(base, D, tau_p0=0.5, weight=w, l_tau=1.0), lv)
                B[f'enc{lv:g}_w{w:g}_D{D}'] = dict(R=f['R']['onset_ms'], L1=f['L1']['onset_ms'], L2=f['L2']['onset_ms'],
                                                   L_minus_R=f['L1']['onset_ms'] - f['R']['onset_ms'])
    res['B_onsets'] = B

    # C: R-stage feasibility over the free photoreceptor box
    vi.FLASH_LEVEL = E
    C = []
    for D in D_FEAS:
        for tau in TAU_GRID:
            for kt in KT_GRID:
                b = tiny(base, D, tau, kt)
                m = vi.measure(b, np.array([0]), {'R': 0, 'L1': 1, 'L2': 2})
                g = vi.train_gates(m, D)
                C.append(dict(D=D, tau_p0_ms=tau, Kt=kt, flash_R_tp=m['flash']['R']['tp_ms'],
                              flash_R_onset=m['flash']['R']['onset_ms'],
                              kernel_tp_BG4=m['BG-4']['R']['kernel_tp_ms'], kernel_tp_BG0=m['BG0']['R']['kernel_tp_ms'],
                              evaluable_BG4=vi.bg_evaluable(m, 'BG-4'), evaluable_BG0=vi.bg_evaluable(m, 'BG0'),
                              G1=g['G1_flash_R_tp'], G3_R=g['G3_R_onset'], G4=g['G4_kernel_tp'], R_STAGE=g['R_STAGE_PASS']))
    res['C_feasibility'] = C
    ev = [c for c in C if c['evaluable_BG4']]
    res['C_summary'] = dict(
        n=len(C), n_evaluable_BG4=len(ev),
        kernel_tp_BG4_range=[min(c['kernel_tp_BG4'] for c in ev), max(c['kernel_tp_BG4'] for c in ev)] if ev else None,
        kernel_tp_BG0_range=[min(c['kernel_tp_BG0'] for c in C), max(c['kernel_tp_BG0'] for c in C)],
        flash_R_tp_range_at_E_sat=[min(c['flash_R_tp'] for c in C), max(c['flash_R_tp'] for c in C)],
        n_G1=sum(c['G1'] for c in C), n_G4_BG4=sum(c['G4'].get('BG-4', False) for c in C),
        n_G4_all=sum(all(c['G4'].values()) for c in C), n_R_STAGE=sum(c['R_STAGE'] for c in C))
    res['wall_s'] = round(time.time() - t0, 1)
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps(dict(E_sat=E, substep_ms=res['substep_ms'], DELAY_MS=DELAY_MS,
                          sat_E={k: v['E_sat'] for k, v in A.items()},
                          mono={k: v['tp_nonincreasing_with_level'] for k, v in A.items()},
                          C=res['C_summary'], wall_s=res['wall_s']), default=float))


if __name__ == '__main__':
    main(sys.argv[1])
