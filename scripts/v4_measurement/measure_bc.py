"""Predeclared measurements (b) and (c), spec §7.10.

A drifting grating delivered to R1-R6 only, eight directions in the retinotopic
lattice plane, on the real MaleCNS graph under LIF v4.  Reports per-population
membrane response (graded cells) and firing rate (spiking cells), the HS
physiology comparison, and the direction-selectivity index per population.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

W = '<redacted-path>/Documents/ChatGPT/flybrain/<redacted-path>/.wt-graded'
sys.path.insert(0, W)
from brainlab.brain import Brain
from brainlab.graded_policy import POLICY_PRIMARY
from brainlab.photoreceptor_io import (PhotoreceptorGratingEncoder,
                                       resolve_photoreceptor_io)

p = argparse.ArgumentParser()
p.add_argument('--directions', type=int, default=8)
p.add_argument('--gray-ms', type=float, default=500.0)
p.add_argument('--grating-ms', type=float, default=1000.0)
p.add_argument('--window-ms', type=float, default=500.0, help='response window: last N ms')
p.add_argument('--step-ms', type=float, default=2.0)
p.add_argument('--policy', default=POLICY_PRIMARY)
p.add_argument('--dynamics', default='v4')
p.add_argument('--backend', default=None)
p.add_argument('--out', default='<redacted-path>/tmp/graded/measure_bc.json')
args = p.parse_args()

io = resolve_photoreceptor_io()
enc = PhotoreceptorGratingEncoder(io)
graph = os.path.join('<redacted-path>/Documents/ChatGPT/flybrain/outputs/brainlab/malecns_v1',
                     'graph.npz')
t0 = time.perf_counter()
kw = {}
if args.dynamics == 'v4':
    kw['graded_policy'] = args.policy
brain = Brain(graph, dynamics=args.dynamics, backend=args.backend, **kw)
load_s = time.perf_counter() - t0
graded = brain.graded.astype(bool) if args.dynamics == 'v4' else np.zeros(brain.n, bool)

pops = dict(io.trace)
pop_graded = {k: bool(graded[v].all()) for k, v in pops.items()}
directions = [i * 360.0 / args.directions for i in range(args.directions)]
n_gray = int(round(args.gray_ms / args.step_ms))
n_grat = int(round(args.grating_ms / args.step_ms))
n_win = int(round(args.window_ms / args.step_ms))

result = dict(
    spec='docs/LIF_DYNAMICS_SPEC.md#7.10',
    dynamics=args.dynamics, graded_policy=args.policy,
    graded_report=brain.graded_report, backend=brain.backend,
    io_map=io.describe(), encoder=enc.describe(),
    protocol=dict(directions=directions, gray_ms=args.gray_ms, grating_ms=args.grating_ms,
                  response_window_ms=args.window_ms, step_ms=args.step_ms,
                  contrast_gray=0.0, contrast_grating=1.0, noise=None, seeds=None),
    graph_load_s=load_s,
    populations={k: dict(n=int(len(v)), graded=pop_graded[k]) for k, v in pops.items()},
    per_direction={},
)

currents = np.zeros(brain.n, np.float32)
hs_nodes = np.concatenate([pops[k] for k in ('HS_L', 'HS_R')])
wall0 = time.perf_counter()
for theta in directions:
    brain.reset_state()
    enc_t = 0.0
    acc = {}

    def window(n_steps, contrast, tag, collect):
        global enc_t
        sums = {k: 0.0 for k in pops}
        spikes = {k: 0 for k in pops}
        vmin = {k: 1e9 for k in pops}
        vmax = {k: -1e9 for k in pops}
        brain_spikes = 0
        max_single = 0
        v_at_floor = 0
        v_at_ceiling = 0
        hs_v = np.zeros(len(hs_nodes))
        for _ in range(n_steps):
            currents.fill(0.0)
            enc.encode(currents, enc_t, theta, contrast)
            counts, _ = brain.step(currents, args.step_ms)
            enc_t += args.step_ms
            if not collect:
                continue
            brain._sync_from_gpu()
            v = brain.v
            brain_spikes += int(counts.sum())
            max_single = max(max_single, int(counts.max()))
            v_at_floor += int((v[graded] <= brain.e_inh_mV + 1e-3).sum())
            v_at_ceiling += int((v[graded] >= -1e-3).sum())
            hs_v += v[hs_nodes]
            for k, idx in pops.items():
                sums[k] += float(v[idx].mean())
                spikes[k] += int(counts[idx].sum())
                vmin[k] = min(vmin[k], float(v[idx].min()))
                vmax[k] = max(vmax[k], float(v[idx].max()))
        if not collect:
            return None
        dur_s = n_steps * args.step_ms / 1000.0
        return dict(
            mean_v_mV={k: sums[k] / n_steps for k in pops},
            min_v_mV=vmin, max_v_mV=vmax,
            rate_hz={k: spikes[k] / dur_s / len(pops[k]) for k in pops},
            spikes={k: spikes[k] for k in pops},
            brain_mean_hz=brain_spikes / dur_s / brain.n,
            brain_max_single_hz=max_single / (args.step_ms / 1000.0),
            graded_at_floor_frac=v_at_floor / (n_steps * max(1, int(graded.sum()))),
            graded_at_ceiling_frac=v_at_ceiling / (n_steps * max(1, int(graded.sum()))),
            hs_v_mV=(hs_v / n_steps).tolist(),
        )

    # gray (mean luminance, contrast 0): settle, then measure the baseline
    window(n_gray - n_win, 0.0, 'settle', False)
    base = window(n_win, 0.0, 'gray', True)
    # grating: settle, then measure the response window
    window(n_grat - n_win, 1.0, 'settle', False)
    resp = window(n_win, 1.0, 'grating', True)
    result['per_direction'][f'{theta:.0f}'] = dict(baseline=base, response=resp)
    el = time.perf_counter() - wall0
    print(f'theta={theta:.0f} done, {el:.1f} s elapsed', flush=True)

result['wall_s'] = time.perf_counter() - wall0
result['simulated_s'] = len(directions) * (args.gray_ms + args.grating_ms) / 1000.0

# --- derived: response, tuning curve, DSI per population -------------------
dsi = {}
for k in pops:
    curve = {}
    for theta in directions:
        d = result['per_direction'][f'{theta:.0f}']
        if pop_graded[k]:
            curve[theta] = d['response']['mean_v_mV'][k] - d['baseline']['mean_v_mV'][k]
        else:
            curve[theta] = d['response']['rate_hz'][k] - d['baseline']['rate_hz'][k]
    pos = {t: max(0.0, v) for t, v in curve.items()}
    pref = max(pos, key=lambda t: pos[t])
    anti = (pref + 180.0) % 360.0
    denom = pos[pref] + pos.get(anti, 0.0)
    dsi[k] = dict(unit='mV' if pop_graded[k] else 'Hz',
                  tuning={f'{t:.0f}': curve[t] for t in directions},
                  preferred_deg=pref, anti_deg=anti,
                  r_pref=pos[pref], r_anti=pos.get(anti, 0.0),
                  magnitude=max(abs(v) for v in curve.values()),
                  dsi=((pos[pref] - pos.get(anti, 0.0)) / denom) if denom > 0 else None)
result['direction_selectivity'] = dsi

with open(args.out, 'w') as fh:
    json.dump(result, fh, indent=2)
print('wrote', args.out)
for k in ('R1-R6_L', 'R1-R6_R', 'L1_L', 'L2_L', 'L3_L', 'Mi1_L', 'Mi9_L', 'Tm1_L', 'Tm9_L',
          'T4a_L', 'T4b_L', 'T4c_L', 'T4d_L', 'T5a_L', 'T5b_L', 'T5c_L', 'T5d_L',
          'HS_L', 'HS_R', 'H2_L', 'VS_L', 'DNa02_L', 'DNa02_R'):
    if k in dsi:
        d = dsi[k]
        print(f"{k:10s} {d['unit']:2s} mag={d['magnitude']:9.4f} pref={d['preferred_deg']:5.0f} "
              f"DSI={d['dsi'] if d['dsi'] is None else round(d['dsi'], 3)}")
