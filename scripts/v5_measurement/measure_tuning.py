"""Predeclared measurement (b)/(c) of docs/LIF_DYNAMICS_SPEC.md §9.9, real graph, GPU.

The v4 photoreceptor protocol (same encoder, same grating, same pins, same eight
lattice-plane directions), with the declared v5 changes to the PROTOCOL only:

* gray: 1,000 ms settle from rest, then a 1,000 ms baseline window (v4 measured its
  baseline in the first 500 ms after reset, i.e. during the settling transient);
* each grating condition starts from the SAME post-gray state (snapshot), runs
  500 ms of grating to pass the onset transient, then a 2,000 ms response window =
  exactly 3 cycles of the 1.5 Hz grating (v4: 500 ms = 0.75 cycles);
* two YAW conditions (CCW / CW seen from above): front-to-back on one eye and
  back-to-front on the other, directions from anatomy (scripts/v5_measurement/
  yaw_axes.py), for the HS sign test and the DNa02 clause of the gate;
* one full-field FLICKER condition (zero spatial frequency, same f, mean, contrast)
  for each population's temporal phase.

Recorded per neuron over each window (on the GPU): mean V, the first Fourier
component of V at the stimulus frequency (sampled every 2 ms step), spike count.
Per population per step: mean V.  Everything is written as raw JSON/NPZ straight
into docs/receipts/v5_raw/ -- never only to a scratch directory.
"""
import argparse
import json
import math
import os
import time

import numpy as np

from brainlab.brain import Brain
from brainlab.graph_identity import dynamics_pin
from brainlab.photoreceptor_io import (PhotoreceptorGratingEncoder,
                                       resolve_photoreceptor_io)

p = argparse.ArgumentParser()
p.add_argument('--dynamics', default='v5')
p.add_argument('--kinetics', default=None)
p.add_argument('--policy', default=None)
p.add_argument('--tag', required=True)
p.add_argument('--gray-settle-ms', type=float, default=1000.0)
p.add_argument('--gray-window-ms', type=float, default=1000.0)
p.add_argument('--grating-settle-ms', type=float, default=500.0)
p.add_argument('--window-ms', type=float, default=2000.0)
p.add_argument('--step-ms', type=float, default=2.0)
p.add_argument('--conditions', default='all', help='comma list or "all"')
p.add_argument('--outdir', default='docs/receipts/v5_raw')
args = p.parse_args()

YAW = json.load(open('docs/receipts/v5_raw/yaw_axes.json'))
FTB = {e: YAW[e]['theta_front_to_back_deg'] for e in ('L', 'R')}
CONDITIONS = {f'dir{int(t)}': ('grating', {'L': float(t), 'R': float(t)}) for t in range(0, 360, 45)}
# CCW seen from above = leftward pattern motion = front-to-back on the LEFT eye
# and back-to-front on the RIGHT eye (Schnell et al. 2010 sign convention).
CONDITIONS['yaw_ccw'] = ('grating', {'L': FTB['L'], 'R': (FTB['R'] + 180.0) % 360.0})
CONDITIONS['yaw_cw'] = ('grating', {'L': (FTB['L'] + 180.0) % 360.0, 'R': FTB['R']})
CONDITIONS['flicker'] = ('flicker', None)
names = list(CONDITIONS) if args.conditions == 'all' else args.conditions.split(',')

io = resolve_photoreceptor_io()
enc = PhotoreceptorGratingEncoder(io)
graph = 'outputs/brainlab/malecns_v1/graph.npz'
kw = {}
if args.dynamics in ('v4', 'v5') and args.policy:
    kw['graded_policy'] = args.policy
if args.dynamics == 'v5' and args.kinetics:
    kw['kinetics'] = args.kinetics
t0 = time.perf_counter()
brain = Brain(graph, dynamics=args.dynamics, backend='cuda', **kw)
load_s = time.perf_counter() - t0
import cupy  # noqa: E402

gpu = brain._gpu
n = brain.n
graded = brain.graded.astype(bool)
pops = dict(io.trace)
pop_names = sorted(pops)
# population-mean operator (n_pop x n), one SpMV per step
import cupyx.scipy.sparse as csp  # noqa: E402
rows, cols, vals = [], [], []
for r, k in enumerate(pop_names):
    idx = pops[k]
    rows += [r] * len(idx)
    cols += list(idx)
    vals += [1.0 / len(idx)] * len(idx)
P = csp.csr_matrix((cupy.asarray(np.array(vals, np.float32)),
                    (cupy.asarray(np.array(rows)), cupy.asarray(np.array(cols)))),
                   shape=(len(pop_names), n))
watch = {k: pops[k] for k in ('HS_L', 'HS_R', 'H2_L', 'H2_R', 'DNa02_L', 'DNa02_R', 'CT1_L', 'CT1_R')
         if k in pops}
watch_idx = cupy.asarray(np.concatenate(list(watch.values())))
f_hz = enc.temporal_frequency_hz
currents = np.zeros(n, np.float32)
graded_d = cupy.asarray(graded)
n_graded = max(1, int(graded.sum()))


def run(n_steps, t_ms, cond, collect):
    """Advance n_steps control steps; return accumulators if collect."""
    kind, dirs = cond
    acc = None
    if collect:
        acc = dict(sum_v=cupy.zeros(n, cupy.float64), sum_c=cupy.zeros(n, cupy.float64),
                   sum_s=cupy.zeros(n, cupy.float64), spikes=np.zeros(n, np.int64),
                   pop_v=[], watch_v=[], floor=0, ceil=0, brain_spikes=0)
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
            v = gpu.d_v.astype(cupy.float64)
            ph = 2 * math.pi * f_hz * t_ms / 1000.0
            acc['sum_v'] += v
            acc['sum_c'] += v * math.cos(ph)
            acc['sum_s'] += v * math.sin(ph)
            acc['spikes'] += counts
            acc['brain_spikes'] += int(counts.sum())
            acc['pop_v'].append(P.dot(gpu.d_v))
            acc['watch_v'].append(gpu.d_v[watch_idx])
            acc['floor'] += int(((gpu.d_v <= brain.e_inh_mV + 1e-3) & graded_d).sum())
            acc['ceil'] += int(((gpu.d_v >= -1e-3) & graded_d).sum())
    return t_ms, acc


TRACE_IDX = np.unique(np.concatenate([pops[k] for k in pop_names]))
TRACE_IDX_D = cupy.asarray(TRACE_IDX)


def finish(acc, n_steps):
    """Per-neuron arrays are kept for the traced populations only (``trace_idx``)."""
    out = dict(mean_v=cupy.asnumpy(acc['sum_v'][TRACE_IDX_D] / n_steps).astype(np.float32),
               f1_re=cupy.asnumpy(2.0 * acc['sum_c'][TRACE_IDX_D] / n_steps).astype(np.float32),
               f1_im=cupy.asnumpy(2.0 * acc['sum_s'][TRACE_IDX_D] / n_steps).astype(np.float32),
               spikes=acc['spikes'][TRACE_IDX].astype(np.int32),
               pop_v=cupy.asnumpy(cupy.stack(acc['pop_v'])).astype(np.float32),
               watch_v=cupy.asnumpy(cupy.stack(acc['watch_v'])).astype(np.float32))
    scal = dict(graded_at_floor_frac=acc['floor'] / (n_steps * n_graded),
                graded_at_ceiling_frac=acc['ceil'] / (n_steps * n_graded),
                brain_mean_hz=acc['brain_spikes'] / (n_steps * args.step_ms / 1000.0) / n)
    return out, scal


os.makedirs(args.outdir, exist_ok=True)
npz = {'trace_idx': TRACE_IDX.astype(np.int64)}
meta = dict(spec='docs/LIF_DYNAMICS_SPEC.md#9.9', tag=args.tag, dynamics=args.dynamics,
            dynamics_pin=dynamics_pin(args.dynamics), backend=brain.backend,
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

wall0 = time.perf_counter()
brain.reset_state()
t_ms, _ = run(n_settle, 0.0, ('gray', None), False)
t_ms, acc = run(n_base, t_ms, ('gray', None), True)
out, scal = finish(acc, n_base)
for k, v in out.items():
    npz[f'gray__{k}'] = v
meta['scalars']['gray'] = scal
meta['timing']['gray_wall_s'] = time.perf_counter() - wall0
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
    # write after EVERY condition so nothing is lost to an interruption
    np.savez_compressed(os.path.join(args.outdir, f'{args.tag}.npz'), **npz)
    meta['conditions_done'] = [c for c in names if f'{c}__mean_v' in npz]
    meta['wall_s'] = time.perf_counter() - wall0
    sim_s = (args.gray_settle_ms + args.gray_window_ms
             + len(meta['conditions_done']) * (args.grating_settle_ms + args.window_ms)) / 1000.0
    meta['simulated_s'] = sim_s
    meta['simulated_s_per_wall_s'] = sim_s / meta['wall_s']
    with open(os.path.join(args.outdir, f'{args.tag}.meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    print(f'{name} done, {time.perf_counter() - wall0:.0f} s elapsed', flush=True)
print('wrote', args.outdir, args.tag)
