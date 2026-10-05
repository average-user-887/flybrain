from _paths import REPO, SCRATCH  # repo root; scratch = $NEUROFLY_V4_SCRATCH or outputs/v4_measurement
import collections
import numpy as np
import pyarrow.feather as f

ROOT = str(REPO)
n = f.read_table(ROOT + '/connectome_data/malecns_v1/normalized/neurons.feather').to_pandas()
ct = n.cell_type.fillna('').to_numpy()
sc = n.superclass.fillna('').to_numpy()
for ty in ['R1-R6', 'R7d', 'R8d', 'L1', 'L2', 'L3', 'L4', 'L5', 'Lai', 'T1', 'C2', 'C3', 'Mi1', 'Mi4', 'Mi9',
           'Tm1', 'Tm2', 'Tm3', 'Tm4', 'Tm9', 'Tm20', 'TmY3', 'Dm9', 'CT1', 'T2', 'T3',
           'T4a', 'T5a', 'LPi2c', 'HSN', 'HSE', 'HSS', 'HST', 'H2', 'VS', 'LC4', 'LPLC2', 'DNa02', 'Y3', 'Pm1', 'Am1']:
    m = ct == ty
    print(f'{ty:8s} n={int(m.sum()):5d} superclass={sorted(set(sc[m]))}')
print()
print('counts by superclass for optic-lobe-ish classes:')
for s in ['ol_intrinsic', 'ol_sensory', 'visual_projection', 'visual_centrifugal', 'visual_projection_tbc']:
    print(f'  {s:22s} {int((sc == s).sum()):6d}')
# LPTC types by name
lptc = [c for c in set(ct) if str(c).startswith(('HS', 'VS', 'H1', 'H2', 'CH', 'Hx'))]
print('\nLPTC-ish types:', sorted(lptc))
for c in sorted(lptc):
    print(' ', c, int((ct == c).sum()), sorted(set(sc[ct == c])))
