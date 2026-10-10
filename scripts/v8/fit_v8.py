"""v8 input timing: fit of R1-R6 phototransduction timing and L1/L2 membrane timing to Drosophila
R/LMC voltage timing (Mansour 2026, Juusola & Hardie 2001).  Implements
qualification/v8/V8_input_prereg_v3.json (refuses another sha256; v1 and v2 are superseded).  Medulla, synapses (release) and
every edge stay at v6 leak40.  The held-out medulla data are never read here.

  python scripts/v8/fit_v8.py --prereg P --prereg-sha256 SHA --base-params B --base-sha256 SHA \\
      --phase1 DIR --run K --smoke-report FILE --out DIR   (run K = (dead-time grid value, start))
  python scripts/v8/fit_v8.py ... --smoke --out DIR       (step 0: one timed evaluation at run 0, no fit;
                                                           writes smoke_v8.json with the BG decidability)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / 'scripts' / 'v6'))
import v8_input as V  # noqa: E402


def load_prereg(path, sha):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise SystemExit('prereg sha256 mismatch; refusing')
    return json.loads(raw)


def free_spec(pre):
    return [(f['type'], f['key'], float(f['lo']), float(f['hi']), f['scale'] == 'log')
            for f in pre['free_parameters']['continuous']]


def from_x(x, spec):
    out = {}
    for xi, (t, k, lo, hi, lg) in zip(x, spec):
        v = math.exp(xi) if lg else float(xi)
        out.setdefault(t, {})[k] = float(min(max(v, lo), hi))
    return out


def base_values(base, spec):
    vals = []
    for t, k, lo, hi, _ in spec:
        if t == 'R1-R6':
            v = base['phototransduction'][k]
        else:
            p = base['types'][t]; ih = p.get('ih', {})
            v = {'tau_m_ms': p.get('tau_m_ms', 20.0), 'ih_g': ih.get('g', 0.0), 'ih_tau_ms': ih.get('tau_ms', 100.0),
                 'ih_vh_mV': ih.get('vh_mV', -60.0)}[k]
        vals.append(float(min(max(v, lo), hi)))
    return vals


def runs(pre, base, spec):
    """[(run_id, dead_time_ms, x0)]: for each grid dead time, start 0 = base values and starts
    1.. = Latin hypercube (fixed seed) in the transformed box."""
    o = pre['optimiser']
    lo = np.array([math.log(a) if lg else a for _, _, a, _, lg in spec])
    hi = np.array([math.log(b) if lg else b for _, _, _, b, lg in spec])
    rng = np.random.default_rng(int(o['lhs_seed']))
    n_extra = int(o['starts_per_dead_time']) - 1
    x_base = np.array([math.log(v) if lg else v for v, (*_, lg) in zip(base_values(base, spec), spec)])
    out, rid = [], 0
    for dt in pre['free_parameters']['dead_time_grid_ms']:
        u = (np.argsort(rng.random((n_extra, len(spec))), axis=0) + rng.random((n_extra, len(spec)))) / max(n_extra, 1)
        for x0 in [x_base] + [lo + ui * (hi - lo) for ui in u]:
            out.append((rid, int(dt), x0)); rid += 1
    return out


def params_file(base, base_sha, prereg_sha, fitted, dead_time, version):
    p = copy.deepcopy(base)
    for k, v in fitted.get('R1-R6', {}).items():
        p['phototransduction'][k] = v
    p['phototransduction']['dead_time_ms'] = int(dead_time)
    keymap = {'ih_g': 'g', 'ih_tau_ms': 'tau_ms', 'ih_vh_mV': 'vh_mV'}
    for t in V.LMC_TYPES:
        for k, v in fitted.get(t, {}).items():
            if k == 'tau_m_ms':
                p['types'][t]['tau_m_ms'] = v
            else:
                p['types'][t]['ih'][keymap[k]] = v
    p.update(version=version, prereg_sha256=prereg_sha, base_params_sha256=base_sha)
    return p


class InputModel:
    def __init__(self, base, bsha, phase1):
        import fit_s2
        fit_s2.PHASE1 = Path(phase1)
        b, ct, light, cl, cc, info = fit_s2.build_cutout(base, bsha)
        self.b, self.light, self.info = b, light, info
        self.idx = {t: np.flatnonzero(ct == t) for t in set(ct.tolist())}
        if any(str(t).startswith(('T4', 'T5')) for t in self.idx):
            raise ValueError('cutout must not contain T4/T5')
        self.cols = {'R': int(cl[0]), 'L1': int(cc['L1'][0]), 'L2': int(cc['L2'][0])}
        self.pt_base = copy.deepcopy(base['phototransduction'])

    def measure(self, fitted, dead_time):
        f = copy.deepcopy(fitted)
        f.setdefault('R1-R6', {})['dead_time_ms'] = dead_time
        V.apply_input_params(self.b, self.idx, self.light, self.pt_base, f)
        return V.measure(self.b, self.light, self.cols)


def smoke(pre, base, bsha, phase1):
    """Step 0 (prereg v2): one timed measurement at run 0's start values; decides, BEFORE any fit,
    which backgrounds are evaluable (pre-written rule: a background whose kernel peak or response
    s.d. is below the floor is NOT EVALUABLE and is dropped from G4 and the objective for every run;
    if BG0 is not evaluable, or the flash response is non-finite, the whole arm is NOT EVALUABLE)."""
    spec = free_spec(pre)
    rid, dead, x0 = runs(pre, base, spec)[0]
    t0 = time.time()
    m = InputModel(base, bsha, phase1)
    t1 = time.time()
    meas = m.measure(from_x(x0, spec), dead)
    t2 = time.time()
    flash_ok = all(np.isfinite([meas['flash'][n][k] for n in meas['flash'] for k in ('onset_ms', 'tp_ms')]))
    ev = {bg: V.bg_evaluable(meas, bg) for bg in V.BACKGROUND}
    arm = 'EVALUABLE' if flash_ok and ev['BG0'] else 'NOT_EVALUABLE'
    return dict(build_s=round(t1 - t0, 2), eval_s=round(t2 - t1, 3), run0_dead_time_ms=dead, measurements=meas,
                background_evaluable=ev, evaluable_backgrounds=[bg for bg in V.BACKGROUND if ev[bg]],
                flash_finite=bool(flash_ok), arm=arm)


def check_smoke_pin(pre, prereg_sha, report_path, repo=REPO):
    """Prereg v3: the step-0 report must match the sha256 committed in the smoke pin file, which
    must name this prereg.  Returns the parsed report; raises SystemExit otherwise."""
    pin = Path(repo) / pre['smoke_pin']['file']
    if not pin.exists():
        raise SystemExit('step 0 smoke pin not committed; no fit')
    pinned = json.loads(pin.read_text())
    raw = Path(report_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pinned.get('smoke_v8_sha256'):
        raise SystemExit('smoke report sha256 differs from the committed pin; refusing')
    if pinned.get('prereg_sha256') != prereg_sha:
        raise SystemExit('smoke pin belongs to another prereg; refusing')
    return json.loads(raw)


def main():
    ap = argparse.ArgumentParser()
    for k in ('--prereg', '--base-params', '--phase1', '--out'):
        ap.add_argument(k, type=Path, required=True)
    ap.add_argument('--prereg-sha256', required=True); ap.add_argument('--base-sha256', required=True)
    ap.add_argument('--run', type=int)
    ap.add_argument('--smoke-report', type=Path)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    pre = load_prereg(a.prereg, a.prereg_sha256)
    from scipy.optimize import minimize
    from brainlab.v6_calibrated import load_params
    if pre['base_params']['sha256'] != a.base_sha256:
        raise SystemExit('base params sha differs from the prereg')
    base, bsha = load_params(a.base_params, a.base_sha256)
    if a.smoke:
        rep = smoke(pre, base, bsha, a.phase1)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / 'smoke_v8.json').write_text(json.dumps(rep, indent=1, sort_keys=True))
        print(json.dumps({k: rep[k] for k in ('eval_s', 'background_evaluable', 'arm')}, indent=1))
        return
    sm = check_smoke_pin(pre, a.prereg_sha256, a.smoke_report)
    if sm['arm'] != 'EVALUABLE':
        raise SystemExit('step 0 declared the arm NOT EVALUABLE; no fit')
    bgs = tuple(sm['evaluable_backgrounds'])
    spec = free_spec(pre)
    rid, dead, x0 = runs(pre, base, spec)[a.run]
    m = InputModel(base, bsha, a.phase1)
    lo = [math.log(l) if lg else l for _, _, l, _, lg in spec]
    hi = [math.log(h) if lg else h for _, _, _, h, lg in spec]
    hist, t0 = [], time.time()

    def f(x):
        c = V.objective(m.measure(from_x(x, spec), dead), bgs); hist.append(c)
        if len(hist) % 50 == 0:
            print(len(hist), round(min(hist), 3), round(time.time() - t0), 's', flush=True)
        return c
    o = pre['optimiser']
    r = minimize(f, x0, method='Powell', bounds=list(zip(lo, hi)),
                 options=dict(maxfev=int(o['maxfev_per_run']), xtol=float(o['xtol']), ftol=float(o['ftol'])))
    fitted = from_x(r.x, spec)
    meas = m.measure(fitted, dead)
    rep = dict(status='completed', run=rid, dead_time_ms=dead, prereg_sha256=a.prereg_sha256, base_params_sha256=bsha,
               cutout=m.info, evals=len(hist), wall_s=round(time.time() - t0, 1), cost=V.objective(meas, bgs),
               measurements=meas, gates=V.train_gates(meas, dead, bgs), backgrounds=list(bgs), fitted=fitted,
               heldout='NOT READ by the fit (heldout_v8.py after freezing)')
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f'fit_run{rid}.json').write_text(json.dumps(rep, indent=1, sort_keys=True))
    (a.out / f'params_v8_run{rid}.json').write_text(json.dumps(
        params_file(base, bsha, a.prereg_sha256, fitted, dead, f'v8-input-timing run {rid} (candidate)'), indent=1, sort_keys=True))
    print(json.dumps(dict(cost=rep['cost'], TRAIN_PASS=rep['gates']['TRAIN_PASS']), indent=1))


if __name__ == '__main__':
    main()
