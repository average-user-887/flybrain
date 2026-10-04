"""POST-HOC diagnostic (not preregistered; cannot change any verdict): where does the
v5 primary's ~8.6x higher no-stimulus spike rate (F4) live?  1,000 ms settle + 1,000 ms
of the gray protocol under v4 and v5 primary, per-neuron spike counts aggregated by
superclass and by receptor class of the spiking neuron.  Writes
docs/receipts/v5_raw/posthoc_rate_locus.json."""
import collections
import json

import numpy as np

from brainlab.brain import Brain
from brainlab.graded_policy import load_cell_classes
from brainlab.photoreceptor_io import PhotoreceptorGratingEncoder, resolve_photoreceptor_io
from brainlab.receptor_kinetics import classes_from_transmitters
from brainlab.transmitter_policy import load_transmitters

io = resolve_photoreceptor_io()
enc = PhotoreceptorGratingEncoder(io)
ct, sc = load_cell_classes()
rc = classes_from_transmitters(load_transmitters())
graph = 'outputs/brainlab/malecns_v1/graph.npz'
out = {}
for label, kw in (('v4', dict(dynamics='v4')), ('v5_primary', dict(dynamics='v5', kinetics='v5-receptor-class'))):
    b = Brain(graph, backend='cuda', **kw)
    cur = np.zeros(b.n, np.float32)
    counts = np.zeros(b.n, np.int64)
    t = 0.0
    for i in range(1000):
        cur.fill(0)
        enc.encode(cur, t, 0.0, 0.0)
        c, _ = b.step(cur, 2.0)
        t += 2.0
        if i >= 500:
            counts += c
    by_sc = collections.defaultdict(lambda: [0, 0])
    by_rc = collections.defaultdict(lambda: [0, 0])
    for s, r, k in zip(sc, rc, counts):
        by_sc[str(s)][0] += int(k)
        by_sc[str(s)][1] += 1
        by_rc[str(r)][0] += int(k)
        by_rc[str(r)][1] += 1
    active = counts > 0
    out[label] = dict(
        brain_mean_hz=float(counts.sum() / b.n),
        spiking_neurons_active=int(active.sum()),
        by_superclass_hz={k: v[0] / v[1] for k, v in sorted(by_sc.items())},
        by_superclass_spikes={k: v[0] for k, v in sorted(by_sc.items())},
        by_receptor_class_of_spiker_hz={k: v[0] / v[1] for k, v in sorted(by_rc.items())},
        rate_percentiles_hz={p: float(np.percentile(counts[counts > 0], p)) if active.any() else 0.0
                             for p in (50, 90, 99, 100)})
    tc = collections.Counter()
    for t_, k in zip(ct, counts):
        if k:
            tc[str(t_)] += int(k)
    out[label]['top_types_by_spikes'] = tc.most_common(15)
    del b
json.dump(out, open('docs/receipts/v5_raw/posthoc_rate_locus.json', 'w'), indent=1)
for k, v in out.items():
    print(k, round(v['brain_mean_hz'], 3), v['spiking_neurons_active'], v['rate_percentiles_hz'])
    print('  ', {s: round(h, 2) for s, h in v['by_superclass_hz'].items() if h > 0.05})
    print('  ', v['top_types_by_spikes'][:8])
