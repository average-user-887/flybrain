"""The STAGE GATE of docs/LIF_DYNAMICS_SPEC.md §10.8, from one raw run of
measure_tuning.py --phase stage.  Pure post-processing; no simulation.

Statistic (declared): per cell, the F1 amplitude of V at the grating frequency,
averaged over the two yaw conditions (yaw_ccw, yaw_cw); per population and eye,
the 90th percentile over cells (mean and median reported beside it).  T4 and T5
pool their four subtypes.  Gate eye: RIGHT (524 lit cartridges; the left eye has
300 and v5 §14.3 showed its per-cell numbers are dominated by unlit cells).

Also reported (not gating): every traced population's statistic for both eyes,
the v4 / v5 / v5-S2 values of the same statistic from docs/receipts/v5_raw, and
the flicker timing (D4 of analyse_tuning.py) for v6b.
"""
import argparse
import json

import numpy as np

# ---- declared, fixed before the lock (spec §10.8) --------------------------
BANDS_MV = {'lamina': (2.0, 25.0), 't4t5': (1.0, 25.0)}
HEALTH = dict(brain_mean_hz=(0.5, 25.0), graded_at_bound_frac_max=0.05,
              dna02_gray_hz=(0.5, 200.0))
GATE_EYE = 'R'
POPS = ['R1-R6', 'L1', 'L2', 'L3', 'L4', 'L5', 'Mi1', 'Tm3', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm4',
        'Tm9', 'CT1', 'T4', 'T5', 'HS', 'H2', 'VS']

p = argparse.ArgumentParser()
p.add_argument('tag')
p.add_argument('--dir', default='docs/receipts/v6a_raw')
args = p.parse_args()

from brainlab.photoreceptor_io import resolve_photoreceptor_io  # noqa: E402

io = resolve_photoreceptor_io()


def members(k, eye):
    if k in ('T4', 'T5'):
        return np.concatenate([io.trace[f'{k}{s}_{eye}'] for s in 'abcd'])
    return np.asarray(io.trace[f'{k}_{eye}'])


def stage_stats(raw):
    pos = {int(g): i for i, g in enumerate(raw['trace_idx'])}
    out = {}
    for eye in 'LR':
        for k in POPS:
            idx = np.array([pos[int(i)] for i in members(k, eye)])
            A = np.mean([np.hypot(raw[f'{c}__f1_re'][idx], raw[f'{c}__f1_im'][idx])
                         for c in ('yaw_ccw', 'yaw_cw')], 0)
            out[f'{k}_{eye}'] = dict(p90=float(np.percentile(A, 90)), mean=float(A.mean()),
                                     median=float(np.median(A)), n=int(len(idx)),
                                     gray_mean_v=float(raw['gray__mean_v'][idx].mean()))
    return out


meta = json.load(open(f'{args.dir}/{args.tag}.meta.json'))
raw = np.load(f'{args.dir}/{args.tag}.npz')
res = dict(tag=args.tag, dynamics=meta['dynamics'],
           kinetics=(meta.get('kinetics_report') or {}).get('kinetics'),
           release=(meta.get('release_report') or {}).get('release'),
           gpu=meta.get('gpu'), declared_bands_mV=BANDS_MV, declared_health=HEALTH,
           gate_eye=GATE_EYE)
res['stage'] = stage_stats(raw)
res['reference'] = {}
for tag in ('v4_protocol9', 'v5_primary', 'v5_upper_S2'):
    with np.load(f'docs/receipts/v5_raw/{tag}.npz') as r:
        res['reference'][tag] = stage_stats(r)

# health (gray window)
g = meta['scalars']['gray']
pos = {int(x): i for i, x in enumerate(raw['trace_idx'])}
dn = {}
base_s = meta['protocol']['gray_window_ms'] / 1000.0
for side in ('L', 'R'):
    idx = np.array([pos[int(i)] for i in io.trace[f'DNa02_{side}']])
    dn[side] = float(raw['gray__spikes'][idx].mean() / base_s)
bound = g['graded_at_floor_frac'] + g['graded_at_ceiling_frac']
health = dict(
    brain_mean_hz=g['brain_mean_hz'],
    brain_mean_ok=HEALTH['brain_mean_hz'][0] <= g['brain_mean_hz'] <= HEALTH['brain_mean_hz'][1],
    graded_at_bound_frac=bound, graded_at_bound_ok=bound <= HEALTH['graded_at_bound_frac_max'],
    dna02_gray_hz=dn,
    dna02_ok=all(HEALTH['dna02_gray_hz'][0] <= x <= HEALTH['dna02_gray_hz'][1] for x in dn.values()),
    release_clipped_at_zero_frac=g.get('release_clipped_at_zero_frac'),
    release_clipped_at_rmax_frac=g.get('release_clipped_at_rmax_frac'),
    other_windows={c: dict(brain_mean_hz=s['brain_mean_hz'],
                           graded_at_bound_frac=s['graded_at_floor_frac'] + s['graded_at_ceiling_frac'],
                           release_clipped_at_zero_frac=s.get('release_clipped_at_zero_frac'),
                           release_clipped_at_rmax_frac=s.get('release_clipped_at_rmax_frac'))
                   for c, s in meta['scalars'].items() if c != 'gray'})
health['passed'] = bool(health['brain_mean_ok'] and health['graded_at_bound_ok'] and health['dna02_ok'])
res['health'] = health


def in_band(x, band):
    return band[0] <= x <= band[1]


st = res['stage']
e = GATE_EYE
lam = {k: st[f'{k}_{e}']['p90'] for k in ('L1', 'L2')}
tt = {k: st[f'{k}_{e}']['p90'] for k in ('T4', 'T5')}
res['gate'] = dict(
    rule='STAGE GATE (spec §10.8): right-eye p90 per-cell F1 of L1 and of L2 inside the lamina band, '
         'AND of T4 (a-d pooled) and of T5 (a-d pooled) inside the T4/T5 band, AND every network-health '
         'criterion',
    lamina=lam, lamina_ok=all(in_band(x, BANDS_MV['lamina']) for x in lam.values()),
    t4t5=tt, t4t5_ok=all(in_band(x, BANDS_MV['t4t5']) for x in tt.values()),
    health_ok=health['passed'])
res['gate']['passed'] = bool(res['gate']['lamina_ok'] and res['gate']['t4t5_ok'] and res['gate']['health_ok'])

with open(f'{args.dir}/{args.tag}.stage.json', 'w') as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res['gate'], indent=1))
print(json.dumps({k: v for k, v in health.items() if k != 'other_windows'}, indent=1))
for k in POPS:
    row = [f"{k:6s}"]
    for eye in 'RL':
        s = st[f'{k}_{eye}']
        r5 = res['reference']['v5_primary'][f'{k}_{eye}']['p90']
        row.append(f"{eye}: p90 {s['p90']:8.3f} mean {s['mean']:7.3f} (v5 p90 {r5:6.3f}) gray {s['gray_mean_v']:7.2f}")
    print('  '.join(row))
