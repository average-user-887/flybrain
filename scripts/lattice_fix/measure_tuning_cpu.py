"""POST-FINDING CORRECTION (2026-10-08): the frozen LIF v5 tuning protocol
(docs/LIF_DYNAMICS_SPEC.md §9.9, scripts/v5_measurement/measure_tuning.py) on CPU,
with the photoreceptor grating encoder's hex lattice selectable.

Not a new preregistration and not a replacement for the published v5 receipts in
docs/receipts/v5_raw/, which stay untouched.  Same graph, pins, dynamics, encoder
amplitude, grating (30 deg, 1.5 Hz, contrast 1, no noise), windows and step as
measure_tuning.py; the only changes are:

* ``--lattice``: ``axial-v1`` (the published, sheared frame) or ``malecns-hex-v2``
  (the corrected MaleCNS frame; brainlab.photoreceptor_io.LATTICE_VERSIONS);
* the yaw directions use the SAME declared anatomical posteriority coefficients
  (c1, c2) from docs/receipts/v5_raw/yaw_axes.json, expressed in the chosen frame by
  the same rule yaw_axes.py uses (theta = atan2 of L^-T c, L the lattice matrix);
  for axial-v1 this reproduces the published 123.13 / 123.94 deg;
* backend CPU (numpy) instead of CuPy; flicker is not rerun by default because a
  zero-spatial-frequency stimulus does not depend on the lattice.

No neural, model, kinetics, policy or gate parameter is changed.
"""
import argparse
import json
import math
import os
import time

import numpy as np

from brainlab.brain import Brain
from brainlab.graph_identity import dynamics_pin
from brainlab.photoreceptor_io import (PhotoreceptorGratingEncoder, lattice_matrix,
                                       resolve_photoreceptor_io)

p = argparse.ArgumentParser()
p.add_argument('--lattice', required=True)
p.add_argument('--tag', required=True)
p.add_argument('--outdir', required=True)
p.add_argument('--dynamics', default='v5')
p.add_argument('--conditions', default='dir0,dir45,dir90,dir135,dir180,dir225,dir270,dir315,yaw_ccw,yaw_cw')
p.add_argument('--gray-settle-ms', type=float, default=1000.0)
p.add_argument('--gray-window-ms', type=float, default=1000.0)
p.add_argument('--grating-settle-ms', type=float, default=500.0)
p.add_argument('--window-ms', type=float, default=2000.0)
p.add_argument('--step-ms', type=float, default=2.0)
args = p.parse_args()

YAW = json.load(open('docs/receipts/v5_raw/yaw_axes.json'))
L = lattice_matrix(args.lattice)
FTB = {}
for e in ('L', 'R'):
    g = np.linalg.inv(L).T @ np.array([YAW[e]['c1'], YAW[e]['c2']])
    FTB[e] = math.degrees(math.atan2(g[1], g[0])) % 360.0
CONDITIONS = {f'dir{int(t)}': ('grating', {'L': float(t), 'R': float(t)}) for t in range(0, 360, 45)}
CONDITIONS['yaw_ccw'] = ('grating', {'L': FTB['L'], 'R': (FTB['R'] + 180.0) % 360.0})
CONDITIONS['yaw_cw'] = ('grating', {'L': (FTB['L'] + 180.0) % 360.0, 'R': FTB['R']})
CONDITIONS['flicker'] = ('flicker', None)
names = args.conditions.split(',')

io = resolve_photoreceptor_io()
enc = PhotoreceptorGratingEncoder(io, lattice=args.lattice)
t0 = time.perf_counter()
brain = Brain('outputs/brainlab/malecns_v1/graph.npz', dynamics=args.dynamics, backend='cpu')
load_s = time.perf_counter() - t0
n = brain.n
graded = brain.graded.astype(bool)
pops = dict(io.trace)
pop_names = sorted(pops)
watch = {k: pops[k] for k in ('HS_L', 'HS_R', 'H2_L', 'H2_R', 'DNa02_L', 'DNa02_R', 'CT1_L', 'CT1_R')
         if k in pops}
watch_idx = np.concatenate(list(watch.values()))
f_hz = enc.temporal_frequency_hz
currents = np.zeros(n, np.float32)
n_graded = max(1, int(graded.sum()))
TRACE_IDX = np.unique(np.concatenate([pops[k] for k in pop_names]))


def run(n_steps, t_ms, cond, collect):
    kind, dirs = cond
    acc = None
    if collect:
        acc = dict(sum_v=np.zeros(n), sum_c=np.zeros(n), sum_s=np.zeros(n),
                   spikes=np.zeros(n, np.int64), pop_v=[], watch_v=[], floor=0, ceil=0,
                   brain_spikes=0)
    for _ in range(n_steps):
        currents.fill(0.0)
        if kind == 'gray':
            enc.encode(currents, t_ms, 0.0, 0.0)
        elif kind == 'grating':
            enc.encode_per_eye(currents, t_ms, dirs, 1.0)
        else:
            enc.encode_flicker(currents, t_ms, 1.0)
        counts, _ = brain.step(currents, args.step_ms)
        t_ms += args.step_ms
        if collect:
            v = brain.v.astype(np.float64)
            ph = 2 * math.pi * f_hz * t_ms / 1000.0
            acc['sum_v'] += v
            acc['sum_c'] += v * math.cos(ph)
            acc['sum_s'] += v * math.sin(ph)
            acc['spikes'] += counts
            acc['brain_spikes'] += int(counts.sum())
            acc['pop_v'].append(np.array([brain.v[pops[k]].mean() for k in pop_names], np.float32))
            acc['watch_v'].append(brain.v[watch_idx].copy())
            acc['floor'] += int(((brain.v <= brain.e_inh_mV + 1e-3) & graded).sum())
            acc['ceil'] += int(((brain.v >= -1e-3) & graded).sum())
    return t_ms, acc


def finish(acc, n_steps):
    out = dict(mean_v=(acc['sum_v'][TRACE_IDX] / n_steps).astype(np.float32),
               f1_re=(2.0 * acc['sum_c'][TRACE_IDX] / n_steps).astype(np.float32),
               f1_im=(2.0 * acc['sum_s'][TRACE_IDX] / n_steps).astype(np.float32),
               spikes=acc['spikes'][TRACE_IDX].astype(np.int32),
               pop_v=np.stack(acc['pop_v']).astype(np.float32),
               watch_v=np.stack(acc['watch_v']).astype(np.float32))
    scal = dict(graded_at_floor_frac=acc['floor'] / (n_steps * n_graded),
                graded_at_ceiling_frac=acc['ceil'] / (n_steps * n_graded),
                brain_mean_hz=acc['brain_spikes'] / (n_steps * args.step_ms / 1000.0) / n)
    return out, scal


os.makedirs(args.outdir, exist_ok=True)
npz = {'trace_idx': TRACE_IDX.astype(np.int64)}
meta = dict(label='POST-FINDING CORRECTION (hex-lattice shear, 2026-10-08); not a preregistered run',
            spec='docs/LIF_DYNAMICS_SPEC.md#9.9', tag=args.tag, lattice=args.lattice,
            dynamics=args.dynamics, dynamics_pin=dynamics_pin(args.dynamics), backend=brain.backend,
            graded_report=brain.graded_report, kinetics_report=brain.kinetics_report,
            io_map=io.describe(), encoder=enc.describe(), yaw_axes=FTB,
            conditions={k: dict(kind=CONDITIONS[k][0], directions=CONDITIONS[k][1]) for k in names},
            protocol=dict(gray_settle_ms=args.gray_settle_ms, gray_window_ms=args.gray_window_ms,
                          grating_settle_ms=args.grating_settle_ms, window_ms=args.window_ms,
                          step_ms=args.step_ms, cycles_in_window=args.window_ms / 1000.0 * f_hz,
                          contrast=1.0, noise=None, seeds=None,
                          f1='2/N * sum V(t_k) exp(-i 2 pi f t_k), t_k = end of each step'),
            populations={k: dict(n=int(len(v)), graded=bool(graded[v].all())) for k, v in pops.items()},
            pop_order=pop_names, watch={k: [int(i) for i in v] for k, v in watch.items()},
            graph_load_s=load_s, scalars={}, timing={})
n_settle = int(round(args.gray_settle_ms / args.step_ms))
n_base = int(round(args.gray_window_ms / args.step_ms))
n_gs = int(round(args.grating_settle_ms / args.step_ms))
n_win = int(round(args.window_ms / args.step_ms))


def save():
    np.savez_compressed(os.path.join(args.outdir, f'{args.tag}.npz'), **npz)
    meta['conditions_done'] = [c for c in names if f'{c}__mean_v' in npz]
    meta['wall_s'] = time.perf_counter() - wall0
    with open(os.path.join(args.outdir, f'{args.tag}.meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)


wall0 = time.perf_counter()
brain.reset_state()
t_ms, _ = run(n_settle, 0.0, ('gray', None), False)
t_ms, acc = run(n_base, t_ms, ('gray', None), True)
out, scal = finish(acc, n_base)
for k, v in out.items():
    npz[f'gray__{k}'] = v
meta['scalars']['gray'] = scal
save()
snap = brain.snapshot_state()
t_gray_end = t_ms
print(f'gray done {time.perf_counter() - wall0:.0f} s', flush=True)
for name in names:
    c0 = time.perf_counter()
    brain.restore_state(snap)
    t_ms, _ = run(n_gs, t_gray_end, CONDITIONS[name], False)
    t_ms, acc = run(n_win, t_ms, CONDITIONS[name], True)
    out, scal = finish(acc, n_win)
    for k, v in out.items():
        npz[f'{name}__{k}'] = v
    meta['scalars'][name] = scal
    meta['timing'][f'{name}_wall_s'] = time.perf_counter() - c0
    save()
    print(f'{name} done, {time.perf_counter() - wall0:.0f} s elapsed', flush=True)
print('wrote', args.outdir, args.tag)
