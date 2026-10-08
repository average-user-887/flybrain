"""Merge condition-split parts of one S3 arm into <tag>.npz/.meta.json (gray must be identical).
  python scripts/v6/merge_parts.py DIR TAG PART_TAG [PART_TAG ...]"""
import json, sys
import numpy as np
d, tag, parts = sys.argv[1], sys.argv[2], sys.argv[3:]
out, meta = {}, None
for pt in parts:
    z = np.load(f'{d}/{pt}.npz'); m = json.load(open(f'{d}/{pt}.meta.json'))
    for k in z.files:
        if k in out:
            assert np.array_equal(out[k], z[k]), f'{k} differs between parts'
        else:
            out[k] = z[k]
    if meta is None:
        meta = m
    else:
        meta['conditions'].update(m['conditions']); meta['scalars'].update(m['scalars'])
        meta['timing'].update(m['timing']); meta['conditions_done'] += m['conditions_done']
meta['merged_from'] = parts; meta['tag'] = tag
np.savez_compressed(f'{d}/{tag}.npz', **out)
json.dump(meta, open(f'{d}/{tag}.meta.json', 'w'), indent=1)
print('merged', tag, sorted(meta['conditions_done']))
