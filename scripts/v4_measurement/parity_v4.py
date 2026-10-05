"""CPU vs GPU parity for the v4 hybrid engine on a small random graph."""
from _paths import REPO, SCRATCH  # repo root; scratch = $NEUROFLY_V4_SCRATCH or outputs/v4_measurement
import json
import os
import sys

import numpy as np

sys.path.insert(0, str(REPO))
from brainlab.brain import Brain

rng = np.random.default_rng(7)
n, k = 600, 12
post = rng.integers(0, n, size=n * k).astype(np.int32)
ptr = np.arange(0, (n + 1) * k, k, dtype=np.int64)
sign = np.where(rng.random(n * k) < 0.7, 1.0, -1.0)
weight = (sign * rng.uniform(0.5, 6.0, n * k)).astype(np.float32)
arrays = dict(ptr=ptr, post=post, weight=weight, ids=np.arange(1, n + 1, dtype=np.int64))
graded = (rng.random(n) < 0.5).astype(np.uint8)
drive = np.zeros(n, np.float32)
drive[rng.choice(n, 60, replace=False)] = 12.0

out = {}
for backend in ('cpu', 'cuda'):
    b = Brain(arrays=dict(arrays), dynamics='v4', graded_policy=graded, backend=backend)
    tot = np.zeros(n, np.int64)
    for _ in range(20):
        c, _ = b.step(drive, 5.0)
        tot += c
    b._sync_from_gpu()
    out[backend] = dict(total_spikes=int(tot.sum()), v=b.v.copy(), counts=tot.copy(),
                        ge=b.g[0].copy(), gi=b.g[1].copy())

cpu, gpu = out['cpu'], out['cuda']
report = dict(
    total_spikes_cpu=cpu['total_spikes'], total_spikes_gpu=gpu['total_spikes'],
    spike_count_agreement=float(np.mean(cpu['counts'] == gpu['counts'])),
    max_abs_v_diff_mV=float(np.max(np.abs(cpu['v'] - gpu['v']))),
    median_abs_v_diff_mV=float(np.median(np.abs(cpu['v'] - gpu['v']))),
    max_abs_ge_diff=float(np.max(np.abs(cpu['ge'] - gpu['ge']))),
    max_abs_gi_diff=float(np.max(np.abs(cpu['gi'] - gpu['gi']))),
    graded_neurons=int(graded.sum()),
)
print(json.dumps(report, indent=2))
