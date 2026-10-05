"""Pre-lock anatomical survey: transmitter labels per cell type, and the
receptor-class composition of the inputs to T4/T5/HS. Anatomy only; no simulation."""
import collections

import numpy as np

from brainlab.graded_policy import load_cell_classes
from brainlab.transmitter_policy import load_transmitters

lab = load_transmitters().astype(str)
ct, sc = load_cell_classes()
g = np.load('outputs/brainlab/malecns_v1/graph.npz')
ptr, post, w = g['ptr'], g['post'], g['weight']
print(collections.Counter(lab).most_common())
for t in ['R1-R6', 'R7d', 'R8d', 'L1', 'L2', 'L3', 'L4', 'L5', 'Lai', 'Mi1', 'Mi4', 'Mi9', 'Tm1', 'Tm2',
          'Tm3', 'Tm4', 'Tm9', 'CT1', 'T4a', 'T5a', 'HSN', 'HSE', 'HSS', 'H2', 'DNa02', 'Dm9', 'C3', 'C2', 'T1']:
    m = ct == t
    print(t, int(m.sum()), collections.Counter(lab[m]).most_common(3))
pre = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
for tgt in ['T4a', 'T5a', 'HSE', 'L1', 'L2', 'Mi9', 'Mi4', 'Mi1', 'DNa02']:
    sel = ct[post] == tgt
    agg = collections.defaultdict(float)
    for p, ww in zip(pre[sel], w[sel]):
        agg[(ct[p], lab[p])] += abs(float(ww))
    tot = sum(agg.values())
    top = sorted(agg.items(), key=lambda x: -x[1])[:12]
    print('INPUTS to', tgt, [(k, round(v / tot, 3)) for k, v in top])
