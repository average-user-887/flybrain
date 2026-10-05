"""Survey the R1-R6 -> lamina -> medulla -> T4/T5 -> HS/H2 -> DNa02 pathway in the pinned graph."""
import collections
import numpy as np
import pyarrow.feather as f

ROOT = '<redacted-path>/Documents/ChatGPT/flybrain'
t = f.read_table(ROOT + '/connectome_data/malecns_v1/normalized/neurons.feather').to_pandas()
ct = t.cell_type.fillna('').to_numpy()
g = np.load(ROOT + '/outputs/brainlab/malecns_v1/graph.npz')
ptr, post, w = g['ptr'], g['post'], g['weight']
types = set(str(c) for c in ct)
groups = {
    'R1-6': ['R1-R6'],
    'L1-5': ['L1', 'L2', 'L3', 'L4', 'L5'],
    'Lai': ['Lai'],
    'T1': ['T1'],
    'C2C3': ['C2', 'C3'],
    'Mi': sorted(c for c in types if c.startswith('Mi')),
    'Tm': sorted(c for c in types if c.startswith('Tm') and not c.startswith('TmY')),
    'TmY': sorted(c for c in types if c.startswith('TmY')),
    'Dm': sorted(c for c in types if c.startswith('Dm')),
    'CT1': ['CT1'],
    'T4': ['T4a', 'T4b', 'T4c', 'T4d'],
    'T5': ['T5a', 'T5b', 'T5c', 'T5d'],
    'LPi': sorted(c for c in types if c.startswith('LPi')),
    'HS': ['HSN', 'HSE', 'HSS', 'HST'],
    'H2': ['H2'],
    'DNa02': ['DNa02'],
}
lab = np.full(len(ct), '', dtype=object)
for k, v in groups.items():
    lab[np.isin(ct, v)] = k
    print(f'{k:6s} n={int(np.isin(ct, v).sum()):6d}')

pre = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
pl = lab[pre]
ql = lab[post]
m = (pl != '') & (ql != '')
cnt = collections.Counter()
wsum = collections.Counter()
for a, b, wt in zip(pl[m], ql[m], w[m]):
    cnt[(a, b)] += 1
    wsum[(a, b)] += abs(float(wt))
print('\ntop group->group by summed |weight|')
for k in sorted(cnt, key=lambda k: -wsum[k])[:40]:
    print(f'{k[0]:6s} -> {k[1]:6s} edges {cnt[k]:8d} sum|w| {wsum[k]:12.1f}')
print('\nR1-6 targets:')
for k in sorted((k for k in cnt if k[0] == 'R1-6'), key=lambda k: -wsum[k]):
    print(f'  -> {k[1]:6s} edges {cnt[k]:6d} sum|w| {wsum[k]:10.1f}')
print('\ninto T4:')
for k in sorted((k for k in cnt if k[1] == 'T4'), key=lambda k: -wsum[k]):
    print(f'  {k[0]:6s} -> edges {cnt[k]:6d} sum|w| {wsum[k]:10.1f}')
print('\ninto T5:')
for k in sorted((k for k in cnt if k[1] == 'T5'), key=lambda k: -wsum[k]):
    print(f'  {k[0]:6s} -> edges {cnt[k]:6d} sum|w| {wsum[k]:10.1f}')
print('\ninto HS:')
for k in sorted((k for k in cnt if k[1] == 'HS'), key=lambda k: -wsum[k]):
    print(f'  {k[0]:6s} -> edges {cnt[k]:6d} sum|w| {wsum[k]:10.1f}')
print('\ninto DNa02:')
for k in sorted((k for k in cnt if k[1] == 'DNa02'), key=lambda k: -wsum[k]):
    print(f'  {k[0]:6s} -> edges {cnt[k]:6d} sum|w| {wsum[k]:10.1f}')
