"""Predeclared a3 (docs/LIF_DYNAMICS_SPEC.md §9.9), real graph, GPU: v5 with the
v4-equivalent table against v4, 200 ms of the gray protocol (photoreceptors at the
mean drive), every state array compared bit for bit.  Also records wall time."""
import json
import time

import numpy as np

from brainlab import receptor_kinetics as rk
from brainlab.brain import Brain
from brainlab.photoreceptor_io import PhotoreceptorGratingEncoder, resolve_photoreceptor_io

io = resolve_photoreceptor_io()
enc = PhotoreceptorGratingEncoder(io)
graph = 'outputs/brainlab/malecns_v1/graph.npz'
states, walls, spikes = {}, {}, {}
for label, kw in (('v4', dict(dynamics='v4')),
                  ('v5_S0', dict(dynamics='v5', kinetics=rk.KINETICS_V4_EQUIVALENT)),
                  ('v4_repeat', dict(dynamics='v4'))):
    b = Brain(graph, backend='cuda', **kw)
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
def cmp(a, b):
    return {k: dict(bit_identical=bool(np.array_equal(states[a][k], states[b][k])),
                    max_abs_diff=float(np.max(np.abs(states[a][k].astype(np.float64)
                                                      - states[b][k].astype(np.float64)))))
            for k in states[a]}


eq = cmp('v4', 'v5_S0')
out = dict(spec='docs/LIF_DYNAMICS_SPEC.md#9.9 a3', simulated_ms=200.0,
           v4_vs_v5_S0=eq, v4_vs_v4_repeat=cmp('v4', 'v4_repeat'),
           spikes=spikes, wall_s=walls,
           F1_triggered=not all(x['bit_identical'] for x in eq.values()) or spikes['v4'] != spikes['v5_S0'],
           note='v4_vs_v4_repeat establishes whether v4 itself is bit-reproducible run to run on '
                'the real graph on this GPU; F1 as declared requires bit identity regardless')
json.dump(out, open('docs/receipts/v5_raw/s0_real_graph.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
