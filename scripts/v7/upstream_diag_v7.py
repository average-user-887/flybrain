"""v7 upstream diagnostic -- DESCRIPTIVE, OUTSIDE THE v7 PREREG: not a fit, not evidence for S3.

On the frozen S2 7-column cutout, for a frozen parameter file (v6 leak40 and the best v7 start),
measure the central-column R1-R6, L1, L2, L3 (and Mi1, Tm3, Tm1, Tm2 for context):
  * flicker impulse response: Behnia noise (seed 1) about mean encoder level M, filter against
    u = c/2 (v7_data.observe_filter): onset latency (first lag |K| >= 10 % of peak), time to
    peak, peak (mV per unit u per ms);
  * step: mean M for 3 s, then all cutout R1-R6 to 2M for 1 s: onset latency (|dV| >= 10 % of
    the peak deflection), time to peak, peak deflection (mV);
at M = 1, 10 (BG-1, the v7 mapping of Behnia's ~250 cd/m2) and 100 (BG0).
Counterfactual probes (in memory only; NO parameter file is changed): L1 release gain x3 on the
best v7 start, to ask whether the L1 gain explains the small ON response.

  python scripts/v7/upstream_diag_v7.py --params-sets name=FILE:SHA,... --phase1 DIR --out FILE
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / 'scripts' / 'v6'))
import v7_data as D  # noqa: E402

CELLS = ('R1-R6', 'L1', 'L2', 'L3', 'Mi1', 'Tm3', 'Tm1', 'Tm2')
SETTLE_MS, STEP_MS = 3000, 1000
LEVELS = (1.0, 10.0, 100.0)


def onset_and_peak(x, frac=0.10):
    x = np.asarray(x, float)
    k = int(np.argmax(np.abs(x)))
    pk = float(x[k])
    above = np.flatnonzero(np.abs(x[:k + 1]) >= frac * abs(pk))
    return dict(onset_ms=float(above[0]) if len(above) else None, time_to_peak_ms=float(k), peak=pk)


class Probe:
    def __init__(self, params, sha, phase1):
        import fit_s2
        fit_s2.PHASE1 = Path(phase1)
        b, ct, light, cl, cc, info = fit_s2.build_cutout(params, sha)
        self.b, self.light = b, light
        self.idx = {t: np.flatnonzero(ct == t) for t in set(ct.tolist())}
        self.cols = {t: int(cc[t][0]) for t in CELLS if t != 'R1-R6'}
        self.cols['R1-R6'] = int(cl[0])                     # a central-column photoreceptor
        self.info = info

    def _run(self, levels):
        b = self.b
        b.reset_state()
        drive = np.zeros(b.n, np.float32)
        out = np.zeros((len(levels), len(CELLS)))
        cols = [self.cols[t] for t in CELLS]
        for k, lv in enumerate(levels):
            drive[self.light] = lv
            b.step(drive, 1.0)
            out[k] = b.v[cols]
        return out

    def flicker(self, m):
        s = D.behnia_noise(D.NOISE_MS, 1)
        V = self._run(np.concatenate([np.full(SETTLE_MS, m), m * (1 + s)]))[SETTLE_MS:]
        res = {}
        for j, t in enumerate(CELLS):
            K = D.observe_filter(s, V[:, j])
            res[t] = dict(onset_and_peak(K), response_sd_mV=D.response_sd(V[:, j]))
        return res

    def step(self, m):
        V = self._run(np.concatenate([np.full(SETTLE_MS, m), np.full(STEP_MS, 2 * m)]))
        res = {}
        for j, t in enumerate(CELLS):
            base = V[SETTLE_MS - 200:SETTLE_MS, j].mean()
            r = onset_and_peak(V[SETTLE_MS:, j] - base)
            r['rest_mV'] = float(base)
            res[t] = r
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params-sets', required=True)
    ap.add_argument('--phase1', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    from brainlab.v6_calibrated import load_params
    out = dict(label='DESCRIPTIVE diagnostic OUTSIDE the v7 prereg; not a fit; not evidence for S3', sets={})
    for item in a.params_sets.split(','):
        name, spec = item.split('=', 1)
        path, sha = spec.rsplit(':', 1)
        params, psha = load_params(path, sha)
        p = Probe(params, psha, a.phase1)
        r = dict(params_sha256=psha, cutout=p.info,
                 flicker={str(m): p.flicker(m) for m in LEVELS}, step={str(m): p.step(m) for m in LEVELS})
        if name.startswith('v7'):
            i = p.idx['L1']; g0 = p.b.rel_gain[i].copy()
            p.b.rel_gain[i] = 3 * g0                         # in-memory counterfactual only
            r['counterfactual_L1_gain_x3'] = dict(flicker_10=p.flicker(10.0), step_10=p.step(10.0))
            p.b.rel_gain[i] = g0
        out['sets'][name] = r
        print(name, 'done', flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
