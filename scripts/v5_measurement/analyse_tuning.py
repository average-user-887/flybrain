"""Derived quantities of docs/LIF_DYNAMICS_SPEC.md §9.9 from one raw run of
measure_tuning.py.  Pure post-processing of the committed raw files; no simulation.

D1  population-mean DC tuning and the v4 DSI statistic (THE GATE STATISTIC, verbatim)
D2  per-cell first-harmonic (F1) direction selectivity (declared diagnostic)
D3  null controls: R1-R6 and L1-L5 under D1 and D2
D4  full-field flicker: each population's response phase / equivalent delay
D5  HS membrane response to yaw CCW / CW, per cell, sign test
D6  DNa02 L-R under yaw CCW / CW (the gate's DNa02 clause) and the v4-style
    eight-direction reversal reading
"""
import argparse
import json
import math

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('tag')
p.add_argument('--dir', default='docs/receipts/v5_raw')
args = p.parse_args()

meta = json.load(open(f'{args.dir}/{args.tag}.meta.json'))
raw = np.load(f'{args.dir}/{args.tag}.npz')
trace_idx = raw['trace_idx']
pos_of = {int(g): i for i, g in enumerate(trace_idx)}
from brainlab.photoreceptor_io import resolve_photoreceptor_io  # noqa: E402

io = resolve_photoreceptor_io()
pops = {k: np.array([pos_of[int(i)] for i in v]) for k, v in io.trace.items()}
graded = {k: meta['populations'][k]['graded'] for k in pops}
step_s = meta['protocol']['step_ms'] / 1000.0
win_s = meta['protocol']['window_ms'] / 1000.0
base_s = meta['protocol']['gray_window_ms'] / 1000.0
DIRS = [f'dir{t}' for t in range(0, 360, 45) if f'dir{t}__mean_v' in raw]
ANG = [int(d[3:]) for d in DIRS]


def cell_rate(cond, dur):
    return raw[f'{cond}__spikes'] / dur


def pop_dc(cond, k):
    """Graded: mean V response minus gray (mV). Spiking: rate minus gray (Hz)."""
    idx = pops[k]
    if graded[k]:
        return float(raw[f'{cond}__mean_v'][idx].mean() - raw['gray__mean_v'][idx].mean())
    return float(cell_rate(cond, win_s)[idx].mean() - cell_rate('gray', base_s)[idx].mean())


def v4_dsi(curve):
    """Spec §7.10 statistic, verbatim: rectify at 0, pref = argmax, anti = pref+180."""
    pos = {t: max(0.0, v) for t, v in curve.items()}
    pref = max(pos, key=lambda t: pos[t])
    anti = (pref + 180) % 360
    den = pos[pref] + pos.get(anti, 0.0)
    return dict(tuning=curve, preferred_deg=pref, r_pref=pos[pref], r_anti=pos.get(anti, 0.0),
                magnitude=max(abs(v) for v in curve.values()),
                dsi=((pos[pref] - pos.get(anti, 0.0)) / den) if den > 0 else None)


res = dict(tag=args.tag, dynamics=meta['dynamics'], kinetics=(meta.get('kinetics_report') or {}).get('kinetics'),
           D1={}, D2={}, D4={}, D5={}, D6={}, instrument={}, scalars=meta['scalars'])

# ---- D1 -------------------------------------------------------------------
if DIRS:
    for k in pops:
        res['D1'][k] = v4_dsi({t: pop_dc(d, k) for t, d in zip(ANG, DIRS)})
        res['D1'][k]['unit'] = 'mV' if graded[k] else 'Hz'

# ---- D2 -------------------------------------------------------------------
if len(DIRS) == 8:
    for k in pops:
        if not graded[k]:
            continue
        idx = pops[k]
        A = np.stack([np.hypot(raw[f'{d}__f1_re'][idx], raw[f'{d}__f1_im'][idx]) for d in DIRS], 1)
        best_d = np.zeros(len(idx))
        pref = np.zeros(len(idx))
        for a in range(4):
            num = A[:, a] - A[:, a + 4]
            den = A[:, a] + A[:, a + 4]
            d = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
            better = np.abs(d) > np.abs(best_d)
            best_d = np.where(better, d, best_d)
            pref = np.where(better, np.where(d >= 0, ANG[a], ANG[a + 4]), pref)
        absd = np.abs(best_d)
        ang = np.radians(pref)
        R = np.hypot(np.mean(np.cos(ang) * absd), np.mean(np.sin(ang) * absd))
        circ_mean = math.degrees(math.atan2(np.mean(np.sin(ang) * absd),
                                            np.mean(np.cos(ang) * absd))) % 360
        mean_curve = A.mean(0)
        res['D2'][k] = dict(
            mean_f1_amplitude_mV={t: float(x) for t, x in zip(ANG, mean_curve)},
            median_cell_abs_dsi_f1=float(np.median(absd)),
            p90_cell_abs_dsi_f1=float(np.percentile(absd, 90)),
            frac_cells_dsi_f1_ge_0_2=float(np.mean(absd >= 0.2)),
            preferred_direction_histogram={t: int(np.sum(pref == t)) for t in ANG},
            weighted_circular_mean_pref_deg=circ_mean,
            weighted_resultant=float(R),
            population_curve_dsi=v4_dsi({t: float(x) for t, x in zip(ANG, mean_curve)})['dsi'],
            n=int(len(idx)))

# ---- instrument / null controls ---------------------------------------------
for k in ('R1-R6_L', 'R1-R6_R', 'L1_L', 'L2_L', 'L1_R', 'L2_R'):
    if k in res['D1']:
        res['instrument'][k] = dict(D1_dsi=res['D1'][k]['dsi'], D1_magnitude=res['D1'][k]['magnitude'],
                                    D2_median_cell_abs_dsi_f1=res['D2'].get(k, {}).get('median_cell_abs_dsi_f1'))
pop_order = meta['pop_order']
gv = raw['gray__pop_v']
half = gv.shape[0] // 2
res['instrument']['gray_drift_mV_second_minus_first_half'] = {
    k: float(gv[half:, i].mean() - gv[:half, i].mean()) for i, k in enumerate(pop_order)}

# ---- D4 flicker ----------------------------------------------------------
if 'flicker__f1_re' in raw:
    for k in pops:
        if not graded[k]:
            continue
        idx = pops[k]
        z = complex(float(raw['flicker__f1_re'][idx].mean()), float(raw['flicker__f1_im'][idx].mean()))
        amp = abs(z)
        # measure_tuning stores f1_im = +2/N sum V sin(wt), i.e. MINUS the imaginary
        # part of the declared F1 = 2/N sum V e^{-iwt}.  With that storage convention
        # V ~ a cos(wt - lag) gives (re, im) = a (cos lag, sin lag), so
        # lag = +atan2(im_stored, re).  (Post-lock correction of a sign error in the
        # first version of this line; the declared definition is unchanged and the
        # amplitudes used by D2/D5 are unaffected.)  Sampling at the end of each 2 ms
        # step against a drive held at its start value adds ~1 deg (~2 ms).
        lag = math.degrees(math.atan2(z.imag, z.real)) % 360
        sign = 'same' if (lag < 90 or lag > 270) else 'inverted'
        lag_mod = lag if sign == 'same' else (lag - 180) % 360
        if lag_mod > 180:
            lag_mod -= 360
        f = meta['encoder']['temporal_frequency_hz']
        res['D4'][k] = dict(amplitude_mV=amp, phase_lag_deg=lag, sign_vs_light=sign,
                            lag_after_sign_deg=lag_mod,
                            equivalent_delay_ms=lag_mod / 360.0 / f * 1000.0,
                            dc_response_mV=pop_dc('flicker', k))

# ---- D5 HS ---------------------------------------------------------------
for cond in ('yaw_ccw', 'yaw_cw'):
    if f'{cond}__mean_v' not in raw:
        continue
    out = {}
    for k in ('HS_L', 'HS_R', 'H2_L', 'H2_R', 'VS_L', 'VS_R'):
        if k in pops:
            idx = pops[k]
            dv = raw[f'{cond}__mean_v'][idx] - raw['gray__mean_v'][idx]
            f1 = np.hypot(raw[f'{cond}__f1_re'][idx], raw[f'{cond}__f1_im'][idx])
            out[k] = dict(per_cell_dc_mV=[float(x) for x in dv], mean_dc_mV=float(dv.mean()),
                          per_cell_f1_mV=[float(x) for x in f1],
                          gray_v_mV=[float(x) for x in raw['gray__mean_v'][idx]])
    res['D5'][cond] = out
if 'yaw_ccw' in res['D5'] and 'yaw_cw' in res['D5']:
    s = {}
    for k in ('HS_L', 'HS_R'):
        ccw = res['D5']['yaw_ccw'][k]['mean_dc_mV']
        cw = res['D5']['yaw_cw'][k]['mean_dc_mV']
        # Left HS prefers front-to-back on the left eye = CCW; right HS prefers CW.
        ftb, btf = (ccw, cw) if k == 'HS_L' else (cw, ccw)
        s[k] = dict(front_to_back_dc_mV=ftb, back_to_front_dc_mV=btf,
                    depolarises_ftb=ftb > 0, hyperpolarises_btf=btf < 0,
                    ftb_minus_btf_mV=ftb - btf)
    res['D5']['sign_test'] = s

# ---- D6 DNa02 -----------------------------------------------------------
def dn(cond, dur):
    r = cell_rate(cond, dur)
    return float(r[pops['DNa02_L']].mean()), float(r[pops['DNa02_R']].mean())


g_l, g_r = dn('gray', base_s)
res['D6']['gray'] = dict(L=g_l, R=g_r)
for cond in DIRS + [c for c in ('yaw_ccw', 'yaw_cw', 'flicker') if f'{c}__spikes' in raw]:
    l, r = dn(cond, win_s)
    res['D6'][cond] = dict(L=l, R=r, L_minus_R=l - r)
if 'yaw_ccw' in res['D6'] and 'yaw_cw' in res['D6']:
    a, b = res['D6']['yaw_ccw']['L_minus_R'], res['D6']['yaw_cw']['L_minus_R']
    res['D6']['yaw_clause'] = dict(
        ccw_L_minus_R=a, cw_L_minus_R=b,
        rule='CCW (leftward) pattern motion must give L-R >= +1 Hz and CW must give L-R <= -1 Hz '
             '(syndirectional, the optomotor sign; DNa02 steers ipsilaterally)',
        passed=bool(a >= 1.0 and b <= -1.0),
        reverses_at_all=bool(a * b < 0))
if len(DIRS) == 8:
    rev = {}
    for t in range(0, 180, 45):
        a = res['D6'][f'dir{t}']['L_minus_R']
        b = res['D6'][f'dir{t + 180}']['L_minus_R']
        rev[f'{t}/{t + 180}'] = dict(a=a, b=b, reverses=bool(a * b < 0 and min(abs(a), abs(b)) >= 1.0))
    res['D6']['eight_direction_reversal_v4_style'] = rev

# ---- gate (verbatim v4 gate; DNa02 clause on the yaw conditions) -----------
t45 = {k: v for k, v in res['D1'].items() if k[:2] in ('T4', 'T5')}
strong = sorted(k for k, v in t45.items() if v['dsi'] is not None and v['dsi'] >= 0.2 and v['magnitude'] >= 0.5)
res['gate'] = dict(
    rule='LIF_DYNAMICS_SPEC §7.10 gate reused verbatim: DSI >= 0.2 in at least one T4/T5 subtype whose '
         'response magnitude is >= 0.5 mV, AND a DNa02 asymmetry |L-R| >= 1 Hz whose sign follows the '
         'stimulus in both directions (§9.9: evaluated on the yaw CCW / CW conditions)',
    t4t5_subtypes_passing=strong,
    t4t5_clause=bool(strong),
    dna02_clause=res['D6'].get('yaw_clause', {}).get('passed'),
    passed=bool(strong) and bool(res['D6'].get('yaw_clause', {}).get('passed')))

with open(f'{args.dir}/{args.tag}.analysis.json', 'w') as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res['gate'], indent=1))
for k in sorted(res['D1']):
    if k[:2] in ('T4', 'T5', 'HS', 'R1', 'L1', 'L2', 'Mi', 'Tm', 'CT', 'DN'):
        d1 = res['D1'][k]
        d2 = res['D2'].get(k, {})
        print(f"{k:9s} D1 mag={d1['magnitude']:8.4f}{d1['unit']} pref={d1['preferred_deg']:4d} dsi={d1['dsi']}  "
              f"D2 med|dsi|={d2.get('median_cell_abs_dsi_f1', float('nan')):.3f} "
              f"frac>=.2={d2.get('frac_cells_dsi_f1_ge_0_2', float('nan')):.3f} "
              f"pref~{d2.get('weighted_circular_mean_pref_deg', float('nan')):.0f}")
