import collections
import sys

import numpy as np
import pyarrow.feather as f

W = '<redacted-path>/Documents/ChatGPT/flybrain/<redacted-path>/.wt-graded'
sys.path.insert(0, W)
from brainlab.transmitter_policy import apply_policy, load_transmitters

ROOT = '<redacted-path>/Documents/ChatGPT/flybrain'
n = f.read_table(ROOT + '/connectome_data/malecns_v1/normalized/neurons.feather').to_pandas()
ct = n.cell_type.fillna('').to_numpy()
g = np.load(ROOT + '/outputs/brainlab/malecns_v1/graph.npz')
ptr, post, w0 = g['ptr'], g['post'], g['weight']
w, _ = apply_policy(ptr, post, w0, load_transmitters())
pre = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))

r16 = ct == 'R1-R6'
m = r16[pre]
print('R1-R6 out-edges:', int(m.sum()), 'positive', int((w[m] > 0).sum()),
      'negative', int((w[m] < 0).sum()), 'zero', int((w[m] == 0).sum()))
lam = np.isin(ct, ['L1', 'L2', 'L3', 'L4', 'L5'])
m2 = m & lam[post]
print('R1-R6 -> L1-L5 edges:', int(m2.sum()), 'all negative:', bool((w[m2] < 0).all()),
      'sum|w| %.1f' % np.abs(w[m2]).sum())

into = r16[post]
c = collections.Counter()
for a, wt in zip(ct[pre[into]], w[into]):
    c[a] += abs(float(wt))
print('\ninputs TO R1-R6, top by sum|w|:')
for k, v in c.most_common(8):
    print(f'  {k or "(untyped)":12s} {v:9.1f}')
print('total in-edges to R1-R6:', int(into.sum()))
