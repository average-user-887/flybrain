import numpy as np
import pyarrow.feather as f

ROOT = '<redacted-path>/Documents/ChatGPT/flybrain'
a = f.read_table(ROOT + '/connectome_data/malecns_v1/annotations.feather')
print([c for c in a.column_names if 'Hex' in c or 'hex' in c])
t = a.to_pandas().drop_duplicates('bodyId')
n = f.read_table(ROOT + '/connectome_data/malecns_v1/normalized/neurons.feather').to_pandas()
j = n.join(t.set_index('bodyId'), on='source_id', rsuffix='_a')
ct = j.cell_type.fillna('')
for ty in ['R1-R6', 'L1', 'L2', 'Mi1', 'Tm3', 'T4a', 'T5a', 'HSE']:
    s = j[ct.eq(ty)]
    h1 = s.assignedOlHex1
    h2 = s.assignedOlHex2
    print(ty, len(s), 'h1 dtype', h1.dtype, 'non-null', h1.notna().sum(), 'sample', list(h1.head(3)), list(h2.head(3)))
s = j[ct.eq('R1-R6')]
h1 = np.array([np.nan if v is None else v for v in s.assignedOlHex1], dtype=object)
print('R1-R6 hex1 unique kinds', set(type(v).__name__ for v in s.assignedOlHex1))
print('R1-R6 instance/rootSide sample')
print(s[['source_id', 'instance', 'rootSide', 'assignedOlHex1', 'assignedOlHex2']].head(10).to_string())
