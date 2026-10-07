"""v6 S1 fit: R1-R6 phototransduction + sigmoidal R1-R6 release (CPU, reduced models).

Implements qualification/v6/S1_fit_spec.json (refuses another sha256).  Targets are
the NON-MOTION numbers pinned in S0; nothing downstream of the lamina is read.

  python scripts/v6/fit_s1.py --spec-sha256 SHA --pathb EVIDENCE_DIR --out DIR
Writes DIR/params_v6_S1.json and DIR/fit_report.json.
"""
from __future__ import annotations

import argparse, hashlib, json, math, sys
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from brainlab.engine import R_MAX_HZ, TAU_SYN_MS, DELAY_MS  # noqa: E402
from brainlab.v6_calibrated import Phototransduction  # noqa: E402

V_REST, E_INH, E_EXC, DT = -52.0, -70.0, 0.0, 0.1
G_UNIT_INH = 1.0 / (V_REST - E_INH)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------- photoreceptor reduced model ----------------
@njit(cache=True)
def _r_trace(G, Ka, Kt, tau_a, tau_p0, n, D, light, tau_m):
    """Scalar twin of Phototransduction + the passive R membrane (checked equal at start-up)."""
    st = np.zeros(n); a = 0.0
    ring = np.zeros(max(D, 1)); cur = 0
    v = V_REST
    out = np.empty(len(light))
    ea = 1.0 - math.exp(-1.0 / tau_a)
    for t in range(len(light)):
        if D > 0:
            k = cur % D; x = ring[k]; ring[k] = light[t]; cur += 1
        else:
            x = light[t]
        tau = tau_p0 / (1.0 + a / Kt)
        al = 1.0 - math.exp(-1.0 / tau)
        prev = x
        for j in range(n):
            st[j] += al * (prev - st[j]); prev = st[j]
        y = st[n - 1]
        a += ea * (y - a)
        gl = G * y / (1.0 + a / Ka)
        gt = 1.0 + gl
        vinf = (V_REST + gl * E_EXC) / gt
        v = vinf + (v - vinf) * math.exp(-1.0 * gt / tau_m)
        v = min(max(v, E_INH), E_EXC)
        out[t] = v
    return out


def r_trace(pp, light, tau_m=20.0):
    """light: array (ms,) of intensity in BG0 units.  Returns V_R (mV) per ms."""
    return _r_trace(pp['G'], pp['Ka'], pp.get('Kt', pp['Ka']), pp['tau_a_ms'], pp['tau_p0_ms'], int(pp['n_stages']),
                    int(pp['dead_time_ms']), np.asarray(light, np.float64), tau_m)


def r_trace_reference(pp, light, tau_m=20.0):
    pt = Phototransduction(pp, 1)
    v = V_REST
    out = np.empty(len(light))
    for t, I in enumerate(light):
        gl = pt.step(np.array([I]))[0]
        gt = 1.0 + gl
        vinf = (V_REST + gl * E_EXC) / gt
        v = vinf + (v - vinf) * math.exp(-1.0 * gt / tau_m)
        v = min(max(v, E_INH), E_EXC)
        out[t] = v
    return out


def pp_from(x, fixed):
    G, Ka, tp0, Kt = (float(v) for v in np.exp(x))
    return dict(G=G, Ka=Ka, tau_p0_ms=tp0, Kt=Kt, **fixed)


def photoreceptor_metrics(pp):
    m = {}
    flash = np.zeros(400); flash[100] = 3.3                  # dark-adapted 1-ms flash, 3.3 x BG0
    v = r_trace(pp, flash); m['T1_dark_flash_peak_mV'] = float(v.max() - V_REST)
    step = np.zeros(3000); step[100:] = 1.0                  # dark -> BG0
    v = r_trace(pp, step); m['T2_BG0_plateau_frac'] = float((v[-200:].mean() - V_REST) / m['T1_dark_flash_peak_mV'])
    m['BG0_plateau_mV'] = float(v[-200:].mean() - V_REST)
    for name, bg in (('T3_tp_BG0_ms', 1.0), ('H1_tp_BG-4_ms', 1e-4)):
        L = np.full(3400, bg); L[3000] = 6.0 * bg                 # 1-ms increment of 5 x background
        base = r_trace(pp, np.full(3400, bg))
        v = r_trace(pp, L) - base
        m[name] = float(np.argmax(v[3000:3200]))
    s1 = np.full(3000, 0.1); v1 = r_trace(pp, s1)
    m['H2_plateau_ratio_BG-1_over_BG0'] = float((v1[-200:].mean() - V_REST) / m['BG0_plateau_mV'])
    return m


# ---------------- lamina reduced model ----------------
def lamina_peak(rel, lam, v_r, tau_m=20.0):
    """v_r: V_R per ms.  Simulate one L cell of the reduced model at 0.1 ms; returns V_L per ms."""
    gain, vh, s = rel
    return _lamina(gain, vh, s, lam['W'], lam['ge_o'], lam['gi_o'], lam['v0'], np.asarray(v_r, np.float64),
                   tau_m, R_MAX_HZ, TAU_SYN_MS, int(round(DELAY_MS / DT)))


@njit(cache=True)
def _lamina(gain, vh, s, W, ge_o, gi_o, v0, v_r, tau_m, rmax, tau_syn, d):
    ag = math.exp(-DT / tau_syn)
    nt = len(v_r) * 10
    x = np.empty(nt)
    for t in range(nt):
        x[t] = gain * rmax * DT * 1e-3 / (1.0 + math.exp(-(v_r[t // 10] - vh) / s))
    g = W * G_UNIT_INH * x[0] / (1 - ag)
    v = v0
    out = np.empty(len(v_r))
    for t in range(nt):
        xi = x[t - d] if t >= d else x[0]
        g = g * ag + W * G_UNIT_INH * xi
        gi = gi_o + g
        gt = 1.0 + ge_o + gi
        vinf = (V_REST + gi * E_INH) / gt
        v = vinf + (v - vinf) * math.exp(-DT * gt / tau_m)
        if t % 10 == 9:
            out[t // 10] = v
    return out


_RCACHE = {}


def _rc(pp, key, light):
    k = (tuple(sorted(pp.items())), key)
    if k not in _RCACHE:
        _RCACHE[k] = r_trace(pp, light)
    return _RCACHE[k]


def lamina_metrics(rel, lam, pp):
    flash = np.zeros(1500); flash[1200] = 3.3
    vf = _rc(pp, 'flash', flash)
    vl = lamina_peak(rel, lam, vf)
    v_dark = float(vl[1100:1200].mean())
    peak = float(vl[1200:1450].min())
    bg = np.full(2500, 0.1); bg[2200:2400] = 0.2
    vs = lamina_peak(rel, lam, _rc(pp, 'step', bg))
    den = v_dark - E_INH
    return dict(V_dark=v_dark, frac=(v_dark - peak) / den if den >= 0.01 else 0.0,
                H3_step_dV=float(vs[2200:2400].mean() - vs[2100:2200].mean()))


def lamina_inputs(pathb, weight_cache):
    """Per-type reduced-model constants from the graph (R1-R6 -> L summed |w|) and the Path B v4 sham."""
    cc = np.load(pathb / 'phase1/cell_columns.npz')
    sham = np.load(pathb / 'phase3/v4/C0_sham.npz')
    wc = np.load(weight_cache)
    pos = {int(n): k for k, n in enumerate(sham['node'])}
    VB, GB = sham['V_B'], sham['gtot_B']
    ct, eye, st, node = cc['cell_type'], cc['eye'], cc['column_state'], cc['node']
    rR = (ct == 'R1-R6') & (eye == 'R')
    v_r_v4 = float(np.median([VB[pos[int(n)]] for n in node[rR]]))
    x_v4 = (v_r_v4 - E_INH) * R_MAX_HZ * DT * 1e-3 / (E_EXC - E_INH)
    ag = math.exp(-DT / TAU_SYN_MS)
    Win = dict(zip(wc['post'].tolist(), wc['w_from_R'].tolist()))
    out = {}
    for t in ('L1', 'L2', 'L3'):
        sel = node[(ct == t) & (eye == 'R') & (st == 'COMPLETE')]
        rows = []
        for n in sel:
            W = Win.get(int(n), 0.0)
            if W <= 0:
                continue
            vb, gb = float(VB[pos[int(n)]]), float(GB[pos[int(n)]])
            gR = W * G_UNIT_INH * x_v4 / (1 - ag)
            gi_tot = (V_REST - vb * gb) / (-E_INH)        # from vb*gb = V_REST + gi_tot*E_INH (E_EXC = 0)
            gi_o = max(gi_tot - gR, 0.0)
            ge_o = max(gb - 1.0 - gi_tot, 0.0)
            rows.append((W, ge_o, gi_o))
        a = np.array(rows)
        med = np.median(a, 0)
        out[t] = dict(n_cells=int(len(a)), W=float(med[0]), ge_o=float(med[1]), gi_o=float(med[2]),
                      v0=V_REST, V_R_v4_baseline=v_r_v4)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--spec-sha256', required=True)
    ap.add_argument('--pathb', type=Path, required=True)
    ap.add_argument('--weights', type=Path, required=True, help='npz from scripts/v6/r_to_l_weights.py')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    if sha(a.spec) != a.spec_sha256:
        raise SystemExit('S1 fit spec sha mismatch; refusing')
    spec = json.loads(a.spec.read_text())
    fixed = spec['phototransduction']['fixed']
    chk = dict(G=7.0, Ka=0.2, Kt=0.7, tau_p0_ms=9.0, **fixed)
    lt0 = np.zeros(600); lt0[50:400] = 1.0; lt0[100] = 3.3
    assert np.allclose(r_trace(chk, lt0), r_trace_reference(chk, lt0), atol=1e-9), "scalar twin differs"
    tg = spec['phototransduction']['train_targets']

    def cost_pt(x):
        m = photoreceptor_metrics(pp_from(x, fixed))
        return sum(((m[k] - v['target']) / v['tol']) ** 2 for k, v in tg.items())

    best = None
    lo, hi = np.log(spec['phototransduction']['bounds_lo']), np.log(spec['phototransduction']['bounds_hi'])
    for x0 in ([math.log(75), math.log(0.007), math.log(10), math.log(0.5)],
               [math.log(20), math.log(0.05), math.log(8), math.log(1.0)],
               [math.log(200), math.log(0.002), math.log(15), math.log(0.1)]):
        r = minimize(cost_pt, x0, method='Nelder-Mead', bounds=list(zip(lo, hi)),
                     options=dict(maxiter=1500, xatol=1e-4, fatol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    xb = best.x
    pp = pp_from(xb, fixed)
    mpt = photoreceptor_metrics(pp)
    print('phototransduction', pp, mpt, flush=True)

    lam = lamina_inputs(a.pathb, a.weights)
    lt = spec['release']['train_targets']

    def cost_rel(y, s):
        c = 0.0
        for t in spec['release']['fit_types']:
            m = lamina_metrics((math.exp(y[0]), y[1], s), lam[t], pp)
            c += ((m['V_dark'] - lt['V_dark']['target']) / lt['V_dark']['tol']) ** 2
            c += ((m['frac'] - lt['frac']['target']) / lt['frac']['tol']) ** 2
        return c / len(spec['release']['fit_types'])

    chosen, trail = None, []
    for s in spec['release']['s_grid_mV_descending']:
        rb = None
        for y0 in ([0.0, -30.0], [2.0, -40.0], [-1.0, -20.0]):
            r = minimize(cost_rel, y0, args=(s,), method='Nelder-Mead', options=dict(maxiter=300))
            if rb is None or r.fun < rb.fun:
                rb = r
        gain = float(np.clip(math.exp(rb.x[0]), *spec['release']['gain_bounds']))
        vh = float(np.clip(rb.x[1], *spec['release']['vh_bounds_mV']))
        trail.append(dict(s_mV=s, gain=gain, vh_mV=vh, cost=float(rb.fun)))
        print('release s', s, trail[-1], flush=True)
        if rb.fun <= spec['release']['accept_cost']:
            chosen = trail[-1]
            break
    out = a.out; out.mkdir(parents=True, exist_ok=True)
    report = dict(spec_sha256=a.spec_sha256, phototransduction=pp, photoreceptor_metrics=mpt, pt_cost=float(best.fun),
                  lamina_reduced_model=lam, release_trail=trail, release_chosen=chosen)
    if chosen is not None:
        report['lamina_metrics'] = {t: lamina_metrics((chosen['gain'], chosen['vh_mV'], chosen['s_mV']), lam[t], pp)
                                    for t in ('L1', 'L2', 'L3')}
    report['held_out'] = {k: dict(value=(mpt.get(k) if k in mpt else None), **v)
                          for k, v in spec['held_out'].items()}
    if chosen is not None:
        h3 = [report['lamina_metrics'][t]['H3_step_dV'] for t in ('L1', 'L2', 'L3')]
        report['held_out']['H3_LMC_sign_on_increment']['value'] = h3
    (out / 'fit_report.json').write_text(json.dumps(report, indent=1))
    if chosen is None:
        print('NO release fit within accept_cost; S1 fit FAILED', flush=True)
        return 1
    params = dict(version='v6-S1', fit_spec_sha256=a.spec_sha256,
                  phototransduction=dict(pp, encoder_to_intensity=spec['encoder_to_intensity']),
                  types={'R1-R6': {'release': dict(kind='sigmoid', gain=chosen['gain'], vh_mV=chosen['vh_mV'],
                                                   s_mV=chosen['s_mV'])}})
    (out / 'params_v6_S1.json').write_text(json.dumps(params, indent=1, sort_keys=True))
    print(json.dumps(report['held_out'], indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
