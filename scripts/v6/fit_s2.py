"""v6 S2 (exploratory): per-type shape/sign fit of 13 lamina/medulla types on a 7-column cutout.

Implements qualification/v6/S2_fit_spec.json (refuses another sha256).  Targets: Borst 2025
summary numbers (calcium, shape and sign only).  No motion data anywhere.

  python scripts/v6/fit_s2.py --spec SPEC --spec-sha256 SHA --base-params P --base-sha256 SHA --out DIR
"""
from __future__ import annotations

import argparse, hashlib, json, math, sys, time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from brainlab.brain import Brain  # noqa: E402
from brainlab.graph_identity import resolve_graph_dir  # noqa: E402
from brainlab.photoreceptor_io import _tables, resolve_photoreceptor_io  # noqa: E402
from brainlab.v6_calibrated import BrainV6, load_params  # noqa: E402

TYPES = ['L1', 'L2', 'L3', 'L4', 'L5', 'Mi1', 'Tm3', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm4', 'Tm9']
RF_SIGN = [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1]
IR_HP = [39.1, 28.8, 0, 38.1, 12.7, 31.8, 26.0, 0, 0, 29.6, 15.3, 24.9, 0]
IR_LP = [3.8, 5.8, 5.4, 2.3, 4.2, 5.4, 2.7, 3.8, 7.7, 4.4, 1.4, 2.4, 10.7]
CENTER = (17, 27)
NEIGH = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1))
SETTLE, PRE, STEP = 1000, 500, 1500
PHASE1 = None                                         # set from --phase1
FIT_TYPES = [t for t in TYPES if t != 'L4']           # L4 has no column assignment (addendum 1)


def lowpass(x, tau):                      # Borst_Fig1.py form, tau in bins
    if tau < 1:
        return x.copy()
    r = np.zeros_like(x); r[0] = x[0]
    for i in range(len(x) - 1):
        r[i + 1] = (x[i] - r[i]) / tau + r[i]
    return r


def normalize_data(x):                    # Medulla_Library.normalize_data
    x = x - x[0]
    m = max(abs(np.nanmax(x)), abs(np.nanmin(x)))
    return x * 0.0 if np.nanmax(x) == np.nanmin(x) else x / m


def targets():
    sig = np.zeros(200); sig[50:] = 1.0
    sig = lowpass(sig, 5); sig = sig / sig.max()
    out = {}
    for t, s, hp, lp in zip(TYPES, RF_SIGN, IR_HP, IR_LP):
        y = lowpass(sig, lp)
        if hp:
            y = y - lowpass(y, hp)
        out[t] = s * normalize_data(y)
    return out


def observe(v_ms):
    """calcium observation model of the spec: 10 ms bins, baseline-subtract, LP 5 bins, normalise."""
    b = v_ms.reshape(200, 10).mean(1)
    r = b - b[:50].mean()
    c = lowpass(r, 5)
    m = np.abs(c).max()
    return c / m if m > 0 else c


def build_cutout(base_params, base_sha):
    gdir, _ = resolve_graph_dir(None)
    full = Brain(str(gdir / 'graph.npz'), dynamics='v4', backend='cpu')
    tab = _tables(None)
    ct = tab.cell_type.fillna('').to_numpy()
    inst = tab.instance.fillna('').to_numpy(); root = tab.rootSide.fillna('').to_numpy()
    eye = np.array([(f'{a}'.rsplit('_', 1)[-1] or f'{b}') for a, b in zip(inst, root)])
    h1, h2 = tab.assignedOlHex1.to_numpy(float), tab.assignedOlHex2.to_numpy(float)
    cols = {CENTER} | {(CENTER[0] + a, CENTER[1] + b) for a, b in NEIGH}
    keep, col_of = [], {}
    pc = np.load(PHASE1 / 'cell_columns.npz')          # Path B column map (spec addendum 1)
    for n, t, e, a1, a2 in zip(pc['node'], pc['cell_type'], pc['eye'], pc['hex1'], pc['hex2']):
        if t in TYPES and e == 'R' and (int(a1), int(a2)) in cols:
            keep.append(int(n)); col_of[int(n)] = (int(a1), int(a2))
    io = resolve_photoreceptor_io()
    for i, c in zip(io.r_nodes['R'].tolist(), [tuple(c) for c in io.r_cartridge['R'].tolist()]):
        if c in cols:
            keep.append(int(i)); col_of[int(i)] = c
    keep = np.array(sorted(set(keep)))
    pos = {n: k for k, n in enumerate(keep)}
    ptr = [0]; post = []; w = []
    for n in keep:
        s, e = full.ptr[n], full.ptr[n + 1]
        for j, wt in zip(full.post[s:e], full.weight[s:e]):
            if int(j) in pos:
                post.append(pos[int(j)]); w.append(wt)
        ptr.append(len(post))
    arrays = dict(ptr=np.array(ptr, np.int64), post=np.array(post, np.int32), weight=np.array(w, np.float32),
                  ids=np.arange(1, len(keep) + 1, dtype=np.int64))
    sub_ct = ct[keep]
    light = np.array([pos[n] for n in keep if sub_ct[pos[n]] == 'R1-R6'])
    center_light = np.array([pos[n] for n in keep if sub_ct[pos[n]] == 'R1-R6' and col_of[n] == CENTER])
    center_cells = {t: np.array([pos[n] for n in keep if sub_ct[pos[n]] == t and col_of[n] == CENTER]) for t in TYPES}
    b = BrainV6(None, base_params, base_sha, light_nodes=light, cell_type=sub_ct, arrays=arrays,
                graded_policy=np.ones(len(keep), np.uint8))
    info = dict(cells=int(len(keep)), edges=int(len(post)), photoreceptors=int(len(light)),
                center_photoreceptors=int(len(center_light)),
                center_cells={t: int(len(v)) for t, v in center_cells.items()},
                weights_untouched=bool(np.array_equal(np.array(w, np.float32), arrays['weight'])))
    return b, sub_ct, light, center_light, center_cells, info


class Model:
    def __init__(self, b, ct, light, center_light, center_cells):
        self.b, self.ct, self.light, self.cl, self.cc = b, ct, light, center_light, center_cells
        self.idx = {t: np.flatnonzero(ct == t) for t in TYPES}
        self.tg = targets()

    def apply(self, p):
        b = self.b
        for t in TYPES:
            b.tau_m[self.idx[t]] = p['tau_m_ms'][t]
        for t in ('L1', 'L2'):
            ih = p['ih'][t]; i = self.idx[t]
            b.gh[i] = ih['g']; b.h_vh[i] = ih['vh_mV']; b.h_k[i] = 5.0; b.h_tau[i] = ih['tau_ms']; b.h_e[i] = -30.0
        for t, e in p.get('e_leak_mV', {}).items():
            b.e_leak[self.idx[t]] = e

    def run(self, p):
        self.apply(p)
        b = self.b
        b.reset_state()
        drive = np.zeros(b.n, np.float32); drive[self.light] = 10.0
        for _ in range(SETTLE):
            b.step(drive, 1.0)
        V = np.zeros((PRE + STEP, b.n))
        for t in range(PRE + STEP):
            d = drive
            if t >= PRE:
                d = drive.copy(); d[self.cl] = 20.0
            b.step(d, 1.0)
            V[t] = b.v
        return V

    def cost(self, p, detail=False):
        V = self.run(p)
        per = {}
        for t in FIT_TYPES:
            c = observe(V[:, self.cc[t]].mean(1))
            tg = self.tg[t]
            per[t] = float(((c - tg) ** 2).sum() / (tg ** 2).sum())
        tot = float(np.mean(list(per.values())))
        if detail:
            raw = {t: float(V[PRE:, self.cc[t]].mean() - V[:PRE, self.cc[t]].mean()) for t in FIT_TYPES}
            return tot, per, raw
        return tot


def unpack(x):
    p = {'tau_m_ms': {t: (20.0 if t == 'L4' else float(np.clip(math.exp(x[k]), 2, 200))) for k, t in enumerate(TYPES)}, 'ih': {}}
    k = len(TYPES)
    for t in ('L1', 'L2'):
        g, vh, th = x[k:k + 3]; k += 3
        p['ih'][t] = dict(g=float(np.clip(g, 0, 3)), vh_mV=float(np.clip(vh, -70, -40)),
                          tau_ms=float(np.clip(math.exp(th), 20, 1000)))
    return p


def main():
    ap = argparse.ArgumentParser()
    for k in ('--spec', '--base-params', '--out'):
        ap.add_argument(k, type=Path, required=True)
    ap.add_argument('--spec-sha256', required=True); ap.add_argument('--base-sha256', required=True)
    ap.add_argument('--maxfev', type=int, default=2500)
    ap.add_argument('--phase1', type=Path, required=True)
    a = ap.parse_args()
    global PHASE1
    PHASE1 = a.phase1
    if hashlib.sha256(a.spec.read_bytes()).hexdigest() != a.spec_sha256:
        raise SystemExit('spec sha mismatch')
    base, bsha = load_params(a.base_params, a.base_sha256)
    b, ct, light, cl, cc, info = build_cutout(base, bsha)
    print(info, flush=True)
    m = Model(b, ct, light, cl, cc)
    x0 = np.array([math.log(20.0)] * len(TYPES) + [0.01, -55.0, math.log(200.0)] * 2)
    t0 = time.time()
    c0, per0, raw0 = m.cost(unpack(x0), detail=True)
    print('default cost', c0, 'eval s', round(time.time() - t0, 2), flush=True)
    lo = [math.log(2)] * len(TYPES) + [0, -70, math.log(20)] * 2
    hi = [math.log(200)] * len(TYPES) + [3, -40, math.log(1000)] * 2
    hist = []

    def f(x):
        c = m.cost(unpack(x)); hist.append(c)
        if len(hist) % 100 == 0:
            print(len(hist), min(hist), flush=True)
        return c
    r = minimize(f, x0, method='Powell', bounds=list(zip(lo, hi)), options=dict(maxfev=a.maxfev, xtol=1e-3, ftol=1e-4))
    p = unpack(r.x)
    c1, per1, raw1 = m.cost(p, detail=True)
    pl = dict(p, e_leak_mV={'L1': -40.0, 'L2': -40.0, 'L3': -40.0})
    c2, per2, raw2 = m.cost(pl, detail=True)
    rep = dict(spec_sha256=a.spec_sha256, base_params_sha256=bsha, cutout=info, evals=len(hist),
               default=dict(cost=c0, per_type=per0, raw_center_dV_mV=raw0),
               fitted=dict(cost=c1, per_type=per1, raw_center_dV_mV=raw1, params=p,
                           shape_match=[t for t in FIT_TYPES if per1[t] <= 0.20]),
               leak40_sensitivity=dict(cost=c2, per_type=per2, raw_center_dV_mV=raw2),
               validation='NOT EVALUABLE (no untouched recordings)')
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'fit_report.json').write_text(json.dumps(rep, indent=1))
    types = json.loads(json.dumps(base['types']))
    for t in TYPES:
        types.setdefault(t, {})['tau_m_ms'] = p['tau_m_ms'][t]
    for t in ('L1', 'L2'):
        types[t]['ih'] = dict(p['ih'][t], k_mV=5.0, E_mV=-30.0)
    out = dict(version='v6-S2-A (EXPLORATORY; inherits rejected S1-A3 baseline)', fit_spec_sha256=a.spec_sha256,
               base_params_sha256=bsha, phototransduction=base['phototransduction'], types=types)
    (a.out / 'params_v6_S2A.json').write_text(json.dumps(out, indent=1, sort_keys=True))
    out2 = json.loads(json.dumps(out)); out2['version'] = 'v6-S2-A-leak40 (SENSITIVITY, non-identifiable, hypothetical)'
    for t in ('L1', 'L2', 'L3'):
        out2['types'][t]['e_leak_mV'] = -40.0
    (a.out / 'params_v6_S2A_leak40.json').write_text(json.dumps(out2, indent=1, sort_keys=True))
    print(json.dumps(dict(default=c0, fitted=c1, leak40=c2, per_type=per1, raw=raw1), indent=1))


if __name__ == '__main__':
    main()
