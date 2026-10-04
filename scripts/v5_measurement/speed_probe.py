"""Pre-lock COST probe only: wall time per simulated second, v4 vs a v5 kernel with
four distinct placeholder time constants (the K=4 worst case).  No stimulus and no
response is read or recorded; the placeholder table is not the declared one."""
import json
import sys
import time

import numpy as np

from brainlab import receptor_kinetics as rk
from brainlab.brain import Brain

rk.KINETICS['speed-probe-placeholder'] = dict(nicotinic=1.0, gaba_a=2.0, glucl=3.0, hiscl=4.0)
graph = 'outputs/brainlab/malecns_v1/graph.npz'
out = {}
for label, kw in (('v4', dict(dynamics='v4')),
                  ('v5_K4', dict(dynamics='v5', kinetics='speed-probe-placeholder'))):
    b = Brain(graph, backend='cuda', **kw)
    z = np.zeros(b.n, np.float32)
    b.step(z, 20.0)
    t = time.perf_counter()
    for _ in range(100):
        b.step(z, 2.0)
    wall = time.perf_counter() - t
    out[label] = dict(sim_s=0.2, wall_s=wall, sim_s_per_wall_s=0.2 / wall)
    del b
print(json.dumps(out, indent=1))
json.dump(out, open('docs/receipts/v5_raw/speed_probe.json', 'w'), indent=1)
