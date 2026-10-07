"""v6 S1 arm A3: adaptive R1-R6 release sigmoid on the carried S1-A2 phototransduction.

Implements qualification/v6/S1_fit_spec_v3.json (refuses another sha256).
  python scripts/v6/fit_s1_release_v3.py --spec SPEC --spec-sha256 SHA --pathb DIR --weights NPZ --out DIR
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fit_s1 as f  # noqa: E402
from brainlab.engine import R_MAX_HZ, TAU_SYN_MS, DELAY_MS  # noqa: E402

DT, V_REST, E_INH = f.DT, f.V_REST, f.E_INH


@njit(cache=True)
def _lamina_ad(gain, vh0, s, kappa, tau_s, W, ge_o, gi_o, v0, v_r, tau_m, rmax, tau_syn, d, g_unit_inh):
    """Reduced L cell; R release with the kernel's adaptive sigmoid (m updated per 0.1 ms tick, starts at rest
    as after a reset).  Mirrors advance_v6 for one presynaptic pool."""
    ag = math.exp(-DT / tau_syn)
    em = 1.0 - math.exp(-DT / tau_s)
    nt = len(v_r) * 10
    x = np.empty(nt)
    m = V_REST
    for t in range(nt):
        vr = v_r[t // 10]
        m += (vr - m) * em
        vh = vh0 + kappa * (m - V_REST)
        x[t] = gain * rmax * DT * 1e-3 / (1.0 + math.exp(-(vr - vh) / s))
    g = W * g_unit_inh * x[0] / (1 - ag)
    v = v0
    out = np.empty(len(v_r))
    for t in range(nt):
        xi = x[t - d] if t >= d else x[0]
        g = g * ag + W * g_unit_inh * xi
        gi = gi_o + g
        gt = 1.0 + ge_o + gi
        vinf = (V_REST + gi * E_INH) / gt
        v = vinf + (v - vinf) * math.exp(-DT * gt / tau_m)
        if t % 10 == 9:
            out[t // 10] = v
    return out


def lam(rel, L, vr):
    gain, vh0, s, kappa, tau_s = rel
    return _lamina_ad(gain, vh0, s, kappa, tau_s, L['W'], L['ge_o'], L['gi_o'], L['v0'], np.asarray(vr, float),
                      20.0, R_MAX_HZ, TAU_SYN_MS, int(round(DELAY_MS / DT)), f.G_UNIT_INH)


_RC = {}


def rtr(pp, key, light):
    if key not in _RC:
        _RC[key] = f.r_trace(pp, light)
    return _RC[key]


def amp(rel, L, pp, B, key):
    light = np.full(3600, B); light[3000:3300] = 1.5 * B
    vr = rtr(pp, key, light)
    vl = lam(rel, L, vr)
    dR = vr[3000:3300].max() - vr[2900:3000].mean()
    dL = vl[2900:3000].mean() - vl[3000:3300].min()
    return float(dL / dR), float(dR), float(dL)


def metrics(rel, L, pp):
    flash = np.zeros(1500); flash[1200] = 3.3
    vl = lam(rel, L, rtr(pp, 'flash', flash))
    v_dark = float(vl[1100:1200].mean()); peak = float(vl[1200:1450].min())
    den = v_dark - E_INH
    a0 = amp(rel, L, pp, 1.0, 'BG0'); a35 = amp(rel, L, pp, 10 ** -3.5, 'BG-3.5')
    bg = np.full(3500, 0.1); bg[3200:3400] = 0.2
    vs = lam(rel, L, rtr(pp, 'BG-1step', bg))
    return dict(V_dark=v_dark, frac=(v_dark - peak) / den if den >= 0.01 else 0.0, amp_BG0=a0[0],
                dR_BG0=a0[1], dL_BG0=a0[2], H4_amp_BG35=a35[0], dR_BG35=a35[1], dL_BG35=a35[2],
                H3_step_dV=float(vs[3200:3400].mean() - vs[3100:3200].mean()),
                L_at_BG1_mV=float(vs[3100:3200].mean()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--spec-sha256', required=True)
    ap.add_argument('--pathb', type=Path, required=True)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    if f.sha(a.spec) != a.spec_sha256:
        raise SystemExit('spec sha mismatch; refusing')
    spec = json.loads(a.spec.read_text())
    P = spec['phototransduction']
    pp = dict(P['carried_params'], **P['fixed'])
    mpt = f.photoreceptor_metrics(pp)
    lamc = f.lamina_inputs(a.pathb, a.weights)
    R = spec['release']; tg = R['train_targets']; kappa, tau_s = R['fixed']['kappa'], R['fixed']['tau_s_ms']
    types = R['fit_types']

    def cost(y, s):
        rel = (math.exp(y[0]), y[1], s, kappa, tau_s)
        c = 0.0
        for t in types:
            m = metrics(rel, lamc[t], pp)
            c += sum(((m[k] - v['target']) / v['tol']) ** 2 for k, v in tg.items())
        return c / len(types)

    trail = []
    lo_g, hi_g = R['gain_bounds']; lo_v, hi_v = R['vh_bounds_mV']
    for s in R['s_grid_mV_descending']:
        best = None
        for y0 in ([math.log(5.0), -40.0], [math.log(20.0), -45.0], [math.log(1.0), -30.0], [math.log(50.0), -50.0]):
            r = minimize(cost, y0, args=(s,), method='Nelder-Mead',
                         bounds=[(math.log(lo_g), math.log(hi_g)), (lo_v, hi_v)], options=dict(maxiter=400))
            if best is None or r.fun < best.fun:
                best = r
        trail.append(dict(s_mV=s, gain=float(math.exp(best.x[0])), vh0_mV=float(best.x[1]), cost=float(best.fun)))
        print(trail[-1], flush=True)
    cmin = min(t['cost'] for t in trail)
    chosen = [t for t in trail if t['cost'] <= cmin + 0.05][0]      # grid is descending: first = largest s
    rel = (chosen['gain'], chosen['vh0_mV'], chosen['s_mV'], kappa, tau_s)
    lm = {t: metrics(rel, lamc[t], pp) for t in ('L1', 'L2', 'L3')}
    ho = dict(H1=(mpt['H1_tp_BG-4_ms'], spec['held_out']['H1_tp_BG-4_ms']['accept']),
              H2=(mpt['H2_plateau_ratio_BG-1_over_BG0'], spec['held_out']['H2_plateau_ratio_BG-1_over_BG0']['accept']),
              H3=([lm[t]['H3_step_dV'] for t in ('L1', 'L2', 'L3')], 'all < 0'),
              H4=([lm[t]['H4_amp_BG35'] for t in ('L1', 'L2', 'L3')], spec['held_out']['H4_amp_BG-3.5']['accept']))
    passes = dict(H1=ho['H1'][1][0] <= ho['H1'][0] <= ho['H1'][1][1],
                  H2=ho['H2'][1][0] <= ho['H2'][0] <= ho['H2'][1][1],
                  H3=all(v < 0 for v in ho['H3'][0]),
                  H4=all(ho['H4'][1][0] <= v <= ho['H4'][1][1] for v in ho['H4'][0]))
    fit_ok = chosen['cost'] <= R['accept_cost']
    verdict = 'PASS' if fit_ok and sum(passes.values()) >= 3 else 'REJECTED'
    rep = dict(spec_sha256=a.spec_sha256, phototransduction=pp, photoreceptor_metrics=mpt, lamina_reduced_model=lamc,
               release_trail=trail, release_chosen=chosen, lamina_metrics=lm, held_out=ho, held_out_pass=passes,
               fit_within_accept=fit_ok, verdict=verdict)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'fit_report.json').write_text(json.dumps(rep, indent=1))
    params = dict(version='v6-S1', fit_spec_sha256=a.spec_sha256, verdict=verdict,
                  phototransduction=dict(pp, encoder_to_intensity=spec['encoder_to_intensity']),
                  types={'R1-R6': {'release': dict(kind='sigmoid', gain=chosen['gain'], vh_mV=chosen['vh0_mV'],
                                                   s_mV=chosen['s_mV'], kappa=kappa, tau_s_ms=tau_s)}})
    (a.out / 'params_v6_S1.json').write_text(json.dumps(params, indent=1, sort_keys=True))
    print(json.dumps(dict(chosen=chosen, lamina=lm, held_out=ho, passes=passes, verdict=verdict), indent=1))


if __name__ == '__main__':
    main()
