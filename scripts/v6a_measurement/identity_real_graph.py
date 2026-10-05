"""Predeclared a3 of docs/LIF_DYNAMICS_SPEC.md §10.9, real graph, GPU: v6a with the
v5-linear release table against v5, for both declared kinetics tables, 200 ms of
the gray protocol (photoreceptors at the mean drive), every state array compared;
plus a v5 repeat, which sets the run-to-run float tolerance v5 itself has on the
real graph (v5 §14.6).  One graph instance at a time."""
import json
import os
import time

import numpy as np

from brainlab import graded_release as gr
from brainlab import receptor_kinetics as rk
from brainlab.brain import Brain
from brainlab.photoreceptor_io import PhotoreceptorGratingEncoder, resolve_photoreceptor_io

import sys
sys.path.insert(0, os.path.dirname(__file__))
from measure_common import gpu_identity  # noqa: E402

io = resolve_photoreceptor_io()
enc = PhotoreceptorGratingEncoder(io)
graph = 'outputs/brainlab/malecns_v1/graph.npz'
RUNS = [('v5_primary', dict(dynamics='v5', kinetics=rk.KINETICS_PRIMARY)),
        ('v6a_v5linear_primary', dict(dynamics='v6a', kinetics=rk.KINETICS_PRIMARY,
                                      release=gr.RELEASE_V5_LINEAR)),
        ('v5_primary_repeat', dict(dynamics='v5', kinetics=rk.KINETICS_PRIMARY)),
        ('v5_S0', dict(dynamics='v5', kinetics=rk.KINETICS_V4_EQUIVALENT)),
        ('v6a_v5linear_S0', dict(dynamics='v6a', kinetics=rk.KINETICS_V4_EQUIVALENT,
                                 release=gr.RELEASE_V5_LINEAR))]
states, walls, spikes = {}, {}, {}
gpu = None
for label, kw in RUNS:
    b = Brain(graph, backend='cuda', **kw)
    if gpu is None:
        gpu = gpu_identity()
    cur = np.zeros(b.n, np.float32)
    total = 0
    t0 = time.perf_counter()
    t = 0.0
    for _ in range(100):
        cur.fill(0)
        enc.encode(cur, t, 0.0, 0.0)
        c, _ = b.step(cur, 2.0)
        total += int(c.sum())
        t += 2.0
    walls[label] = time.perf_counter() - t0
    b._sync_from_gpu()
    states[label] = {k: getattr(b, k).copy() for k in ('v', 'g', 'refractory', 'rel_ring')}
    spikes[label] = total
    del b
    import cupy
    cupy.get_default_memory_pool().free_all_blocks()
    print(label, total, round(walls[label], 1), flush=True)


def cmp(a, b):
    return {k: dict(bit_identical=bool(np.array_equal(states[a][k], states[b][k])),
                    max_abs_diff=float(np.max(np.abs(states[a][k].astype(np.float64)
                                                      - states[b][k].astype(np.float64)))))
            for k in states[a]}


out = dict(spec='docs/LIF_DYNAMICS_SPEC.md#10.9 a3', simulated_ms=200.0, gpu=gpu,
           v5_primary_vs_v6a_v5linear=cmp('v5_primary', 'v6a_v5linear_primary'),
           v5_primary_vs_v5_repeat=cmp('v5_primary', 'v5_primary_repeat'),
           v5_S0_vs_v6a_v5linear_S0=cmp('v5_S0', 'v6a_v5linear_S0'),
           spikes=spikes, wall_s=walls)
tol = max(out['v5_primary_vs_v5_repeat']['v']['max_abs_diff'], 1e-4)
out['criterion'] = ('v6a with the v5-linear table reproduces v5: identical spike counts and '
                    'max |dV| no larger than max(v5 run-to-run difference, 1e-4 mV)')
out['tolerance_mV'] = tol
out['passed'] = bool(spikes['v5_primary'] == spikes['v6a_v5linear_primary']
                     and spikes['v5_S0'] == spikes['v6a_v5linear_S0']
                     and out['v5_primary_vs_v6a_v5linear']['v']['max_abs_diff'] <= tol
                     and out['v5_S0_vs_v6a_v5linear_S0']['v']['max_abs_diff'] <= tol)
json.dump(out, open('docs/receipts/v6a_raw/identity_real_graph.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
