"""Predeclared real-graph measurements of docs/LIF_DYNAMICS_SPEC.md §10.9 (GPU).

The v5 protocol of §9.9, unchanged (same encoder, grating, pins, settled gray,
2,000 ms = 3-cycle response window, eight lattice directions, yaw CCW/CW from
anatomy, flicker), run under v6a.  It is split into two invocations that share
ONE gray state, so the stage gate is cheap and the motion gate re-uses it:

  --phase stage   gray settle + baseline, then yaw_ccw, yaw_cw, flicker; the
                  post-gray state is saved (outside git, it is a state, not a
                  measurement) for the motion phase;
  --phase motion  restores that state and runs the eight lattice directions
                  dir0 ... dir315, appending to the same raw files.

Recorded per neuron over each window (on the GPU), exactly as v5: mean V, the
first Fourier component of V at the stimulus frequency (sampled every 2 ms
step), spike count; per population per step, mean V.  Added for v6a: per step,
the fraction of graded cells of each release class whose release is clipped at
0 or at r_max.  Raw JSON/NPZ go straight into docs/receipts/v6a_raw/.
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

import sys
sys.path.insert(0, os.path.dirname(__file__))
from measure_common import gpu_identity  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--dynamics', default='v6a')
p.add_argument('--kinetics', default=None)
p.add_argument('--release', default=None)
p.add_argument('--tag', required=True)
p.add_argument('--phase', choices=('stage', 'motion', 'all'), required=True)
p.add_argument('--gray-settle-ms', type=float, default=1000.0)
p.add_argument('--gray-window-ms', type=float, default=1000.0)
p.add_argument('--grating-settle-ms', type=float, default=500.0)
p.add_argument('--window-ms', type=float, default=2000.0)
p.add_argument('--step-ms', type=float, default=2.0)
p.add_argument('--outdir', default='docs/receipts/v6a_raw')
p.add_argument('--statedir', default='/tmp/v6a_state')
args = p.parse_args()

YAW = json.load(open('docs/receipts/v5_raw/yaw_axes.json'))
FTB = {e: YAW[e]['theta_front_to_back_deg'] for e in ('L', 'R')}
CONDITIONS = {f'dir{int(t)}': ('grating', {'L': float(t), 'R': float(t)}) for t in range(0, 360, 45)}
CONDITIONS['yaw_ccw'] = ('grating', {'L': FTB['L'], 'R': (FTB['R'] + 180.0) % 360.0})
CONDITIONS['yaw_cw'] = ('grating', {'L': (FTB['L'] + 180.0) % 360.0, 'R': FTB['R']})
CONDITIONS['flicker'] = ('flicker', None)
STAGE = ['yaw_ccw', 'yaw_cw', 'flicker']
MOTION = [f'dir{t}' for t in range(0, 360, 45)]
names = {'stage': STAGE, 'motion': MOTION, 'all': STAGE + MOTION}[args.phase]


io = resolve_photoreceptor_io()
enc = PhotoreceptorGratingEncoder(io)
graph = 'outputs/brainlab/malecns_v1/graph.npz'
kw = {}
if args.kinetics:
    kw['kinetics'] = args.kinetics
if args.release:
    kw['release'] = args.release
t0 = time.perf_counter()
brain = Brain(graph, dynamics=args.dynamics, backend='cuda', **kw)
load_s = time.perf_counter() - t0
import cupy  # noqa: E402
import cupyx.scipy.sparse as csp  # noqa: E402

gpu = brain._gpu
n = brain.n
graded = brain.graded.astype(bool)
pops = dict(io.trace)
pop_names = sorted(pops)
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
# release classes (v6a): masks for the clip statistics
if args.dynamics == 'v6a':
    from brainlab import graded_release as gr
    rel_cls = brain._rel['classes']
    cls_masks = {c: cupy.asarray(graded & (rel_cls == c)) for c in gr.CLASSES}
else:
    cls_masks = {'all_graded': graded_d}
cls_n = {c: max(1, int(m.sum())) for c, m in cls_masks.items()}


def run(n_steps, t_ms, cond, collect):
    kind, dirs = cond
    acc = None
    if collect:
        acc = dict(sum_v=cupy.zeros(n, cupy.float64), sum_c=cupy.zeros(n, cupy.float64),
                   sum_s=cupy.zeros(n, cupy.float64), spikes=np.zeros(n, np.int64),
                   pop_v=[], watch_v=[], floor=0, ceil=0, brain_spikes=0,
                   clip0={c: 0 for c in cls_masks}, clipmax={c: 0 for c in cls_masks})
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
            if args.dynamics == 'v6a':
                x = (gpu.d_v - gpu.d_rel_v0) * gpu.d_rel_k
                lo = x <= 0
                hi = x >= gpu.rel_max
                for c, m in cls_masks.items():
                    acc['clip0'][c] += int((lo & m).sum())
                    acc['clipmax'][c] += int((hi & m).sum())
    return t_ms, acc


TRACE_IDX = np.unique(np.concatenate([pops[k] for k in pop_names]))
TRACE_IDX_D = cupy.asarray(TRACE_IDX)


def finish(acc, n_steps):
    out = dict(mean_v=cupy.asnumpy(acc['sum_v'][TRACE_IDX_D] / n_steps).astype(np.float32),
               f1_re=cupy.asnumpy(2.0 * acc['sum_c'][TRACE_IDX_D] / n_steps).astype(np.float32),
               f1_im=cupy.asnumpy(2.0 * acc['sum_s'][TRACE_IDX_D] / n_steps).astype(np.float32),
               spikes=acc['spikes'][TRACE_IDX].astype(np.int32),
               pop_v=cupy.asnumpy(cupy.stack(acc['pop_v'])).astype(np.float32),
               watch_v=cupy.asnumpy(cupy.stack(acc['watch_v'])).astype(np.float32))
    scal = dict(graded_at_floor_frac=acc['floor'] / (n_steps * n_graded),
                graded_at_ceiling_frac=acc['ceil'] / (n_steps * n_graded),
                brain_mean_hz=acc['brain_spikes'] / (n_steps * args.step_ms / 1000.0) / n,
                release_clipped_at_zero_frac={c: acc['clip0'][c] / (n_steps * cls_n[c]) for c in cls_masks},
                release_clipped_at_rmax_frac={c: acc['clipmax'][c] / (n_steps * cls_n[c]) for c in cls_masks},
                graded_class_n=cls_n)
    return out, scal


os.makedirs(args.outdir, exist_ok=True)
os.makedirs(args.statedir, exist_ok=True)
npz_path = os.path.join(args.outdir, f'{args.tag}.npz')
meta_path = os.path.join(args.outdir, f'{args.tag}.meta.json')
state_path = os.path.join(args.statedir, f'{args.tag}.gray_state.npz')
n_settle = int(round(args.gray_settle_ms / args.step_ms))
n_base = int(round(args.gray_window_ms / args.step_ms))
n_gs = int(round(args.grating_settle_ms / args.step_ms))
n_win = int(round(args.window_ms / args.step_ms))
wall0 = time.perf_counter()

if args.phase in ('stage', 'all'):
    npz = {'trace_idx': TRACE_IDX.astype(np.int64)}
    meta = dict(spec='docs/LIF_DYNAMICS_SPEC.md#10.9', tag=args.tag, dynamics=args.dynamics,
                dynamics_pin=dynamics_pin(args.dynamics), backend=brain.backend,
                gpu={args.phase: gpu_identity()},
                graded_report=brain.graded_report, kinetics_report=brain.kinetics_report,
                release_report=getattr(brain, 'release_report', None),
                io_map=io.describe(), encoder=enc.describe(), yaw_axes=FTB,
                conditions={k: dict(kind=CONDITIONS[k][0], directions=CONDITIONS[k][1])
                            for k in STAGE + MOTION},
                protocol=dict(gray_settle_ms=args.gray_settle_ms, gray_window_ms=args.gray_window_ms,
                              grating_settle_ms=args.grating_settle_ms, window_ms=args.window_ms,
                              step_ms=args.step_ms, cycles_in_window=args.window_ms / 1000.0 * f_hz,
                              contrast=1.0, noise=None, seeds=None,
                              f1='2/N * sum V(t_k) exp(-i 2 pi f t_k), t_k = end of each step'),
                populations={k: dict(n=int(len(v)), graded=bool(graded[v].all())) for k, v in pops.items()},
                pop_order=pop_names, watch={k: [int(i) for i in v] for k, v in watch.items()},
                graph_load_s=load_s, scalars={}, timing={})
    brain.reset_state()
    t_ms, _ = run(n_settle, 0.0, ('gray', None), False)
    t_ms, acc = run(n_base, t_ms, ('gray', None), True)
    out, scal = finish(acc, n_base)
    for k, v in out.items():
        npz[f'gray__{k}'] = v
    meta['scalars']['gray'] = scal
    meta['timing']['gray_wall_s'] = time.perf_counter() - wall0
    snap = brain.snapshot_state()
    np.savez(state_path, t_gray_end=t_ms, **{k: v for k, v in snap.items()
                                              if isinstance(v, np.ndarray)},
             **{f'scalar__{k}': np.array(v) for k, v in snap.items() if not isinstance(v, np.ndarray)})
    meta['gray_state_saved_to'] = state_path
    np.savez_compressed(npz_path, **npz)
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=1)
    print(f'gray done {time.perf_counter() - wall0:.0f} s', json.dumps(scal), flush=True)
    t_gray_end = t_ms
else:
    meta = json.load(open(meta_path))
    meta['gpu'][args.phase] = gpu_identity()
    with np.load(npz_path) as z:
        npz = {k: z[k] for k in z.files}
    with np.load(state_path) as z:
        snap = {k: z[k] for k in z.files if not k.startswith('scalar__') and k != 't_gray_end'}
        for k in z.files:
            if k.startswith('scalar__'):
                val = z[k].item()
                snap[k[8:]] = val
        t_gray_end = float(z['t_gray_end'])

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
    np.savez_compressed(npz_path, **npz)
    meta['conditions_done'] = [c for c in STAGE + MOTION if f'{c}__mean_v' in npz]
    meta.setdefault('wall_s_by_phase', {})[args.phase] = time.perf_counter() - wall0
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=1)
    print(f'{name} done, {time.perf_counter() - wall0:.0f} s elapsed', json.dumps(scal), flush=True)
print('wrote', args.outdir, args.tag)
