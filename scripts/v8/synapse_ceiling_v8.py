"""v8 preregistered DESCRIPTIVE test of the L1 -> Mi1/Tm3 transfer shortfall (no fit, no gate on
any arm; parameter files are never changed; the L1-silencing step is an in-memory counterfactual).

On the frozen S2 cutout for a given parameter file:
  (a) steady-state transfer: encoder 0, 0.1, 1, 10, 100, 1000 (3 s each from reset): central
      L1, Mi1, Tm3 voltage; slope dV(Mi1)/dV(L1) between neighbouring levels;
  (b) Behnia Fig. 2 protocol: 3 s dark, 1 s flash at encoder 20: Mi1/Tm3 onset peak (mV);
  (c) disinhibition CEILING (v2): 3 s dark, then L1 release gain set to 0 in memory (= L1 clamped
      at E_inh, its linear release is zero) for 1 s; the TRANSIENT peak Mi1/Tm3 deflection in that
      second, compared like with like with Behnia's transient Fig. 2 onset peak.
Preregistered reading (v2, audit 1 item 6 / audit 2 item 6): first the SIGN - a ceiling <= 0 means
silencing L1 does not depolarise the cell (net excitatory or no L1 path) and the disinhibition
reading is NOT APPLICABLE; otherwise the threshold is the digitised Fig. 2 onset peak minus twice its
digitisation uncertainty (Mi1 30.2 - 2 x 0.92 = 28.4 mV; Tm3 24.7 - 2 x 0.79 = 23.1 mV).  Below it:
'SUGGESTIVE that L1 disinhibition alone cannot reach the measured ON amplitude in this model'
(biological cell-to-cell spread is unknown, so never 'structural').

  python scripts/v8/synapse_ceiling_v8.py --params F --params-sha256 SHA --phase1 DIR --out FILE
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO, HERE, REPO / 'scripts' / 'v6'):
    sys.path.insert(0, str(p))
LEVELS = (0.0, 0.1, 1.0, 10.0, 100.0, 1000.0)
BEHNIA_ONSET_MV = {'Mi1': (30.2, 0.92), 'Tm3': (24.7, 0.79)}     # digitised Fig. 2, (value, uncertainty)


def threshold(cell):
    v, u = BEHNIA_ONSET_MV[cell]
    return v - 2 * u


def reading(cell, ceiling_mV):
    if not ceiling_mV > 0:
        return 'NOT APPLICABLE: silencing L1 does not depolarise ' + cell + ' (no net disinhibition)'
    if ceiling_mV < threshold(cell):
        return 'BELOW: suggestive that L1 disinhibition alone cannot reach the measured ' + cell + ' ON amplitude'
    return 'REACHABLE: the measured ' + cell + ' ON amplitude is within the disinhibition ceiling'


def _run(b, light, cols, levels):
    drive = np.zeros(b.n, np.float32); out = np.zeros((len(levels), len(cols)))
    for k, lv in enumerate(levels):
        drive[light] = lv; b.step(drive, 1.0); out[k] = b.v[cols]
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ('--params', '--phase1', '--out'):
        ap.add_argument(k, type=Path, required=True)
    ap.add_argument('--params-sha256', required=True)
    a = ap.parse_args()
    import fit_s2
    from brainlab.v6_calibrated import load_params
    fit_s2.PHASE1 = a.phase1
    params, psha = load_params(a.params, a.params_sha256)
    b, ct, light, _cl, cc, info = fit_s2.build_cutout(params, psha)
    names = ('L1', 'Mi1', 'Tm3')
    cols = [int(cc[t][0]) for t in names]
    ss = []
    for lv in LEVELS:
        b.reset_state(); ss.append(_run(b, light, cols, np.full(3000, lv))[-1])
    ss = np.array(ss)
    slope = [float((ss[k + 1, 1] - ss[k, 1]) / (ss[k + 1, 0] - ss[k, 0])) if ss[k + 1, 0] != ss[k, 0] else None
             for k in range(len(LEVELS) - 1)]
    b.reset_state()
    pre = _run(b, light, cols, np.zeros(3000))[-1]
    fl = _run(b, light, cols, np.full(1000, 20.0)) - pre
    b.reset_state()
    pre = _run(b, light, cols, np.zeros(3000))[-1]
    i = np.flatnonzero(ct == 'L1'); g0 = b.rel_gain[i].copy(); b.rel_gain[i] = 0.0
    tr = _run(b, light, cols, np.zeros(1000)) - pre
    ce = np.array([tr[:, j][int(np.argmax(np.abs(tr[:, j])))] for j in range(len(names))])   # signed transient peak
    b.rel_gain[i] = g0
    rep = dict(label='DESCRIPTIVE preregistered synapse-transfer test; no fit, no gate', params_sha256=psha,
               steady_state_mV={str(lv): dict(zip(names, map(float, r))) for lv, r in zip(LEVELS, ss)},
               slope_dMi1_dL1=slope,
               flash_onset_peak_mV={n: float(fl[:, j][int(np.argmax(np.abs(fl[:, j])))]) for j, n in enumerate(names)},
               disinhibition_ceiling_mV={n: float(ce[j]) for j, n in enumerate(names)},
               behnia_fig2_onset_mV={c: v for c, (v, _u) in BEHNIA_ONSET_MV.items()},
               thresholds_mV={c: threshold(c) for c in BEHNIA_ONSET_MV},
               reading={c: reading(c, float(ce[names.index(c)])) for c in BEHNIA_ONSET_MV})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True))
    print(json.dumps({k: rep[k] for k in ('flash_onset_peak_mV', 'disinhibition_ceiling_mV', 'reading')}, indent=1))


if __name__ == '__main__':
    main()
