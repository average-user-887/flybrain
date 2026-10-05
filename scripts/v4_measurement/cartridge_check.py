"""Can each R1-R6 be given a retinotopic position from the hex coordinate of its lamina targets?"""
from _paths import REPO, SCRATCH  # repo root; scratch = $NEUROFLY_V4_SCRATCH or outputs/v4_measurement
import collections
import numpy as np
import pyarrow.feather as f

ROOT = str(REPO)
n = f.read_table(ROOT + '/connectome_data/malecns_v1/normalized/neurons.feather').to_pandas()
a = f.read_table(ROOT + '/connectome_data/malecns_v1/annotations.feather').to_pandas().drop_duplicates('bodyId')
j = n.join(a.set_index('bodyId'), on='source_id', rsuffix='_a')
ct = j.cell_type.fillna('').to_numpy()
h1 = j.assignedOlHex1.to_numpy(dtype=float)
h2 = j.assignedOlHex2.to_numpy(dtype=float)
side = j.somaSide.fillna('?').to_numpy()
root = j.rootSide.fillna('?').to_numpy()

print('types with hex coverage > 50%:')
have = ~np.isnan(h1)
by = collections.Counter()
tot = collections.Counter()
for c, hv in zip(ct, have):
    tot[c] += 1
    by[c] += int(hv)
rows = [(c, by[c], tot[c]) for c in tot if tot[c] > 20 and by[c] / tot[c] > 0.5]
rows.sort(key=lambda r: -r[1])
print(len(rows), 'types;', sum(r[1] for r in rows), 'neurons with hex')
print([r[0] for r in rows][:60])

g = np.load(ROOT + '/outputs/brainlab/malecns_v1/graph.npz')
ptr, post = g['ptr'], g['post']
lam = np.isin(ct, ['L1', 'L2', 'L3', 'L4', 'L5'])
r16 = np.flatnonzero(ct == 'R1-R6')
ok = 0
amb = 0
none = 0
assigned = {}
for i in r16:
    tg = post[ptr[i]:ptr[i + 1]]
    tg = tg[lam[tg] & ~np.isnan(h1[tg])]
    if len(tg) == 0:
        none += 1
        continue
    coords = collections.Counter(zip(h1[tg].astype(int), h2[tg].astype(int)))
    top, cnt = coords.most_common(1)[0]
    if len(coords) == 1:
        ok += 1
    else:
        amb += 1
    assigned[i] = top
print(f'R1-R6: single-cartridge {ok}, multi-cartridge(majority used) {amb}, unassignable {none}')
# spread of cartridges per side
for s in ('L', 'R'):
    idx = [i for i in assigned if root[i] == s]
    c1 = np.array([assigned[i][0] for i in idx])
    c2 = np.array([assigned[i][1] for i in idx])
    print(s, 'n', len(idx), 'hex1', c1.min(), c1.max(), 'hex2', c2.min(), c2.max(),
          'distinct cartridges', len(set(zip(c1.tolist(), c2.tolist()))),
          'R per cartridge mean %.2f' % (len(idx) / max(1, len(set(zip(c1.tolist(), c2.tolist()))))))
# do L1 hexes differ between sides?
for s in ('L', 'R'):
    m = (ct == 'L1') & (side == s) & have
    print('L1', s, 'n', m.sum(), 'hex1', np.nanmin(h1[m]), np.nanmax(h1[m]), 'hex2', np.nanmin(h2[m]), np.nanmax(h2[m]))
