"""Analytic check on the first visual synapse: what the tonic baseline delivers."""
import sys

import numpy as np
import pyarrow.feather as f

W = '<redacted-path>/Documents/ChatGPT/flybrain/<redacted-path>/.wt-graded'
sys.path.insert(0, W)
from brainlab.engine import (E_EXC_MV, E_INH_MV, G_UNIT_INH_V3, R_MAX_HZ,
                             TAU_SYN_MS, V_REST_MV)
from brainlab.transmitter_policy import apply_policy, load_transmitters

ROOT = '<redacted-path>/Documents/ChatGPT/flybrain'
n = f.read_table(ROOT + '/connectome_data/malecns_v1/normalized/neurons.feather').to_pandas()
ct = n.cell_type.fillna('').to_numpy()
g = np.load(ROOT + '/outputs/brainlab/malecns_v1/graph.npz')
ptr, post, w0 = g['ptr'], g['post'], g['weight']
w, _ = apply_policy(ptr, post, w0, load_transmitters())

r16 = np.flatnonzero(ct == 'R1-R6')
decay = np.exp(-0.1 / TAU_SYN_MS)
r_rest = R_MAX_HZ * (V_REST_MV - E_INH_MV) / (E_EXC_MV - E_INH_MV)
print(f'r(V_rest) = {r_rest:.3f} s^-1; steady conductance per unit weight = '
      f'{G_UNIT_INH_V3 * r_rest * 1e-4 / (1 - decay):.6f} leak units')

for target in ('L1', 'L2', 'L3', 'L4', 'L5'):
    tg = np.flatnonzero(ct == target)
    tset = np.zeros(len(ct), bool)
    tset[tg] = True
    tot = np.zeros(len(ct))
    for i in r16:
        sl = slice(ptr[i], ptr[i + 1])
        m = tset[post[sl]]
        if m.any():
            np.add.at(tot, post[sl][m], np.abs(w[sl][m]))
    per = tot[tg]
    got = per[per > 0]
    gi = G_UNIT_INH_V3 * r_rest * 1e-4 / (1 - decay) * got
    # Steady membrane with only this tonic inhibition: V = (V_rest + gi*E_inh)/(1+gi)
    v = (V_REST_MV + gi * E_INH_MV) / (1 + gi)
    print(f'{target}: {len(got)}/{len(tg)} cells receive R1-R6 input; sum|w| per cell '
          f'mean {got.mean():.1f} (min {got.min():.1f} max {got.max():.1f}); '
          f'tonic g_inh mean {gi.mean():.3f}; V with that inhibition alone mean '
          f'{v.mean():.2f} mV (min {v.min():.2f})')
