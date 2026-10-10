"""v7 medulla timing: per-type fit of Mi1/Tm3/Tm1/Tm2 timing to Behnia 2014 Fig. 3 filters (mV).

Implements qualification/v7/V7_medulla_prereg.json (refuses another sha256).  Only TRAINING data
are read (v7_data.load_training); the held-out set is evaluated separately by heldout_v7.py
after the fitted parameters are frozen.  No grating, no direction selectivity, no T4/T5, no
behaviour: the cutout contains no T4/T5 cell.  No edge or base weight is changed; every free
parameter is a per-type quantity that BrainV6 already reads from a parameter file.

  python scripts/v7/fit_v7.py --prereg P --prereg-sha256 SHA --base-params B --base-sha256 SHA \\
      --training-dir DIR --phase1 DIR --start K --out DIR
(one process per start; the prereg fixes the starts, the budget and the selection rule)
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
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
import v7_data as D  # noqa: E402

SETTLE_MS = 3000                     # gray (mean level) before the noise


def load_prereg(path, sha):
    raw = Path(path).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != sha:
        raise SystemExit(f'prereg sha256 {got} != {sha}; refusing')
    return json.loads(raw)


def free_spec(prereg):
    """[(type, key, lo, hi, log)] in prereg order."""
    return [(f['type'], f['key'], float(f['lo']), float(f['hi']), f['scale'] == 'log')
            for f in prereg['free_parameters']['list']]


def to_x(values, spec):
    return np.array([math.log(v) if lg else v for v, (_, _, _, _, lg) in zip(values, spec)])


def from_x(x, spec):
    """Transformed vector -> {type: {key: value}}, clipped to the prereg bounds."""
    out = {}
    for xi, (t, k, lo, hi, lg) in zip(x, spec):
        v = math.exp(xi) if lg else float(xi)
        out.setdefault(t, {})[k] = float(min(max(v, lo), hi))
    return out


def base_values(base, spec):
    vals = []
    for t, k, lo, hi, _ in spec:
        p = base['types'].get(t, {})
        if k == 'release_gain':
            v = (p.get('release') or {}).get('gain', 1.0)
        elif k == 'e_leak_mV':
            v = p.get('e_leak_mV', -52.0)
        else:
            v = p.get(k, 20.0)
        vals.append(float(min(max(v, lo), hi)))
    return vals


def starts(prereg, base, spec):
    """Start 0 = base values; starts 1..n = Latin hypercube in the transformed box (fixed seed)."""
    s = prereg['optimiser']
    lo = np.array([math.log(a) if lg else a for _, _, a, _, lg in spec])
    hi = np.array([math.log(b) if lg else b for _, _, _, b, lg in spec])
    rng = np.random.default_rng(int(s['lhs_seed']))
    n = int(s['n_starts']) - 1
    u = (np.argsort(rng.random((n, len(spec))), axis=0) + rng.random((n, len(spec)))) / max(n, 1)
    return [to_x(base_values(base, spec), spec)] + [lo + ui * (hi - lo) for ui in u]


def apply_params(b, idx, fitted):
    """Write per-type values into a BrainV6.  Only tau_m, e_leak and linear release gain."""
    for t, kv in fitted.items():
        i = idx[t]
        for k, v in kv.items():
            if k == 'tau_m_ms':
                b.tau_m[i] = v
            elif k == 'e_leak_mV':
                b.e_leak[i] = v
            elif k == 'release_gain':
                if np.any(b.rel_s[i] > 0):
                    raise ValueError(f'{t}: release gain is fitted only for linear-release types')
                b.rel_gain[i] = v
            else:
                raise ValueError(f'unknown free parameter {k}')


def params_file(base, base_sha, prereg_sha, fitted, version):
    """The frozen v7 parameter file: base types unchanged except the fitted keys."""
    types = copy.deepcopy(base['types'])
    for t, kv in fitted.items():
        e = types.setdefault(t, {})
        for k, v in kv.items():
            if k == 'release_gain':
                e['release'] = dict(e.get('release') or {}, kind='linear', gain=v)
            else:
                e[k] = v
    return dict(version=version, prereg_sha256=prereg_sha, base_params_sha256=base_sha,
                phototransduction=copy.deepcopy(base['phototransduction']), types=types)


class NoiseModel:
    """Full-field Behnia noise on every cutout R1-R6; somatic V of the central-column cell."""

    def __init__(self, b, ct, light, center_cells, seed):
        self.b, self.light = b, light
        self.idx = {t: np.flatnonzero(ct == t) for t in set(ct.tolist())}
        self.cc = {t: center_cells[t] for t in D.RECORDED}
        for t in D.RECORDED:
            if len(self.cc[t]) != 1:
                raise ValueError(f'need exactly one central {t}, got {len(self.cc[t])}')
        if any(str(t).startswith(('T4', 'T5')) for t in set(ct.tolist())):
            raise ValueError('the fit cutout must not contain T4/T5')
        self.s = D.behnia_noise(D.NOISE_MS, seed)
        self.enc = D.contrast_to_encoder(self.s).astype(np.float32)

    def run(self, fitted):
        b = self.b
        apply_params(b, self.idx, fitted)
        b.reset_state()
        drive = np.zeros(b.n, np.float32); drive[self.light] = 10.0
        for _ in range(SETTLE_MS):
            b.step(drive, 1.0)
        V = np.zeros((D.NOISE_MS, len(D.RECORDED)))
        cols = np.array([self.cc[t][0] for t in D.RECORDED])
        for k in range(D.NOISE_MS):
            drive[self.light] = self.enc[k]
            b.step(drive, 1.0)
            V[k] = b.v[cols]
        return V

    def filters(self, fitted):
        V = self.run(fitted)
        return {t: D.observe_filter(self.s, V[:, j]) for j, t in enumerate(D.RECORDED)}


def main():
    ap = argparse.ArgumentParser()
    for k in ('--prereg', '--base-params', '--training-dir', '--phase1', '--out'):
        ap.add_argument(k, type=Path, required=True)
    ap.add_argument('--prereg-sha256', required=True); ap.add_argument('--base-sha256', required=True)
    ap.add_argument('--start', type=int, required=True)
    a = ap.parse_args()
    pre = load_prereg(a.prereg, a.prereg_sha256)
    from scipy.optimize import minimize
    from brainlab.v6_calibrated import load_params
    sys.path.insert(0, str(REPO / 'scripts' / 'v6'))
    import fit_s2                                            # the frozen S2 7-column cutout builder
    fit_s2.PHASE1 = a.phase1
    if pre['base_params']['sha256'] != a.base_sha256:
        raise SystemExit('base params sha differs from the prereg')
    base, bsha = load_params(a.base_params, a.base_sha256)
    data = D.load_training(a.training_dir, pre['data']['training']['files'])
    spec = free_spec(pre)
    b, ct, light, _cl, cc, info = fit_s2.build_cutout(base, bsha)
    m = NoiseModel(b, ct, light, cc, int(pre['stimulus']['training_seed']))
    x0 = starts(pre, base, spec)[a.start]
    lo = [math.log(l) if lg else l for _, _, l, _, lg in spec]
    hi = [math.log(h) if lg else h for _, _, _, h, lg in spec]
    hist = []
    t0 = time.time()

    def f(x):
        c = D.objective(m.filters(from_x(x, spec)), data)[0]
        hist.append(c)
        if len(hist) % 50 == 0:
            print(len(hist), round(min(hist), 4), round(time.time() - t0), 's', flush=True)
        return c
    opt = pre['optimiser']
    r = minimize(f, x0, method='Powell', bounds=list(zip(lo, hi)),
                 options=dict(maxfev=int(opt['maxfev_per_start']), xtol=float(opt['xtol']), ftol=float(opt['ftol'])))
    fitted = from_x(r.x, spec)
    K = m.filters(fitted)
    cost, per, nrmse = D.objective(K, data)
    a.out.mkdir(parents=True, exist_ok=True)
    rep = dict(prereg_sha256=a.prereg_sha256, base_params_sha256=bsha, start=a.start, cutout=info,
               evals=len(hist), wall_s=round(time.time() - t0, 1), cost=cost, per_type=per, fitted=fitted,
               gates=D.train_gates(K, data), model_filters={t: K[t].tolist() for t in D.RECORDED},
               heldout='NOT READ by the fit (heldout_v7.py, after freezing)')
    (a.out / f'fit_start{a.start}.json').write_text(json.dumps(rep, indent=1, sort_keys=True))
    out = params_file(base, bsha, a.prereg_sha256, fitted, f'v7-medulla-timing start {a.start} (candidate)')
    (a.out / f'params_v7_start{a.start}.json').write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(dict(cost=cost, gates=rep['gates']['TRAIN_PASS'], nrmse=nrmse), indent=1))


def select(reports):
    """Prereg selection rule: lowest training cost; ties -> lowest start index."""
    return min(reports, key=lambda r: (r['cost'], r['start']))


if __name__ == '__main__':
    main()
