"""Per-cell diagnostic trace: where does the photoreceptor signal die?

Reported alongside the predeclared population means of measure_bc.py.  For each
population this records, per cell, the DC membrane response (mean over the
response window minus the gray window) and the temporal modulation depth
(standard deviation over the window), so a population whose grating response is
phase-modulated -- which cancels in a population mean -- is still visible.
"""
import argparse
import json
import sys

import numpy as np

W = '<redacted-path>/Documents/ChatGPT/flybrain/<redacted-path>/.wt-graded'
sys.path.insert(0, W)
from brainlab.brain import Brain
from brainlab.graded_policy import POLICY_PRIMARY
from brainlab.photoreceptor_io import (PhotoreceptorGratingEncoder,
                                       resolve_photoreceptor_io)

p = argparse.ArgumentParser()
p.add_argument('--dynamics', default='v4')
p.add_argument('--policy', default=POLICY_PRIMARY)
p.add_argument('--directions', type=float, nargs='+', default=[0.0, 180.0])
p.add_argument('--gray-ms', type=float, default=500.0)
p.add_argument('--grating-ms', type=float, default=1000.0)
p.add_argument('--window-ms', type=float, default=500.0)
p.add_argument('--step-ms', type=float, default=2.0)
p.add_argument('--out', required=True)
args = p.parse_args()

io = resolve_photoreceptor_io()
enc = PhotoreceptorGratingEncoder(io)
kw = dict(graded_policy=args.policy) if args.dynamics == 'v4' else {}
brain = Brain('<redacted-path>/Documents/ChatGPT/flybrain/outputs/brainlab/malecns_v1/graph.npz',
              dynamics=args.dynamics, **kw)
pops = dict(io.trace)
graded = brain.graded.astype(bool) if args.dynamics == 'v4' else np.zeros(brain.n, bool)
currents = np.zeros(brain.n, np.float32)
n_win = int(round(args.window_ms / args.step_ms))

out = dict(dynamics=args.dynamics, policy=args.policy, backend=brain.backend,
           io_map_sha256=io.sha256, directions=args.directions, per_direction={})


def run_window(theta, n_steps, contrast, t0, collect):
    s1 = np.zeros(brain.n)
    s2 = np.zeros(brain.n)
    spikes = np.zeros(brain.n, np.int64)
    t = t0
    for _ in range(n_steps):
        currents.fill(0.0)
        enc.encode(currents, t, theta, contrast)
        counts, _ = brain.step(currents, args.step_ms)
        t += args.step_ms
        if collect:
            brain._sync_from_gpu()
            v = brain.v.astype(np.float64)
            s1 += v
            s2 += v * v
            spikes += counts
    if not collect:
        return t, None
    mean = s1 / n_steps
    std = np.sqrt(np.maximum(0.0, s2 / n_steps - mean * mean))
    return t, dict(mean=mean, std=std, spikes=spikes)


for theta in args.directions:
    brain.reset_state()
    t = 0.0
    t, _ = run_window(theta, int(round(args.gray_ms / args.step_ms)) - n_win, 0.0, t, False)
    t, base = run_window(theta, n_win, 0.0, t, True)
    t, _ = run_window(theta, int(round(args.grating_ms / args.step_ms)) - n_win, 1.0, t, False)
    t, resp = run_window(theta, n_win, 1.0, t, True)
    rows = {}
    for k, idx in pops.items():
        dc = resp['mean'][idx] - base['mean'][idx]
        rows[k] = dict(
            n=int(len(idx)), graded=bool(graded[idx].all()),
            v_gray_mean=float(base['mean'][idx].mean()),
            v_grating_mean=float(resp['mean'][idx].mean()),
            dc_mean_mV=float(dc.mean()),
            dc_abs_mean_mV=float(np.abs(dc).mean()),
            dc_p95_abs_mV=float(np.percentile(np.abs(dc), 95)),
            modulation_gray_mV=float(base['std'][idx].mean()),
            modulation_grating_mV=float(resp['std'][idx].mean()),
            modulation_gain=float(resp['std'][idx].mean() - base['std'][idx].mean()),
            spikes_gray=int(base['spikes'][idx].sum()),
            spikes_grating=int(resp['spikes'][idx].sum()),
        )
    dur = args.window_ms / 1000.0
    rows['__brain__'] = dict(
        n=int(brain.n), graded_neurons=int(graded.sum()),
        mean_hz_gray=float(base['spikes'].sum() / dur / brain.n),
        mean_hz_grating=float(resp['spikes'].sum() / dur / brain.n),
        max_single_hz_gray=float(base['spikes'].max() / dur),
        max_single_hz_grating=float(resp['spikes'].max() / dur),
        active_neurons_grating=int((resp['spikes'] > 0).sum()),
        graded_at_floor_frac=float((resp['mean'][graded] <= brain.e_inh_mV + 1e-2).mean())
        if graded.any() else 0.0,
        graded_at_ceiling_frac=float((resp['mean'][graded] >= -1e-2).mean())
        if graded.any() else 0.0,
        v_min_mV=float(resp['mean'].min()), v_max_mV=float(resp['mean'].max()))
    print('  brain:', json.dumps(rows['__brain__']), flush=True)
    out['per_direction'][f'{theta:.0f}'] = rows
    print(f'theta {theta:.0f} done', flush=True)

with open(args.out, 'w') as fh:
    json.dump(out, fh, indent=2)
print('wrote', args.out)
order = ['R1-R6', 'L1', 'L2', 'L3', 'L4', 'L5', 'Mi1', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm3',
         'Tm4', 'Tm9', 'CT1', 'T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d',
         'HS', 'H2', 'VS', 'DNa02', 'PFL3']
th = f'{args.directions[0]:.0f}'
print(f'\n{"pop":10s} {"eye":3s} {"V_gray":>8s} {"DC":>9s} {"|DC|":>8s} {"mod_gray":>9s} '
      f'{"mod_grat":>9s} {"spk_grat":>8s}')
for name in order:
    for eye in ('L', 'R'):
        k = f'{name}_{eye}'
        if k in out['per_direction'][th]:
            r = out['per_direction'][th][k]
            print(f"{name:10s} {eye:3s} {r['v_gray_mean']:8.3f} {r['dc_mean_mV']:+9.4f} "
                  f"{r['dc_abs_mean_mV']:8.4f} {r['modulation_gray_mV']:9.4f} "
                  f"{r['modulation_grating_mV']:9.4f} {r['spikes_grating']:8d}")
