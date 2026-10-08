"""S3 analysis-side guard (added after the prereg, per Astra's in-flight review; the prereg is unchanged).

Refuses (exit 2, NOT EVALUABLE) unless, for every arm: all 10 preregistered conditions are present;
meta pins (prereg sha, params sha for v6, protocol, encoder, lattice, yaw axes, dynamics pin) agree
across the parts of a split arm and across arms where they must; trace_idx and every array shape
agree; every value is finite.  For split v6 arms it checks the PARTS before the merge.
  python scripts/v6/s3_guard.py DIR"""
import json, sys
from pathlib import Path
import numpy as np

COND = ['dir0', 'dir45', 'dir90', 'dir135', 'dir180', 'dir225', 'dir270', 'dir315', 'yaw_ccw', 'yaw_cw']
FIELDS = ['f1_im', 'f1_re', 'mean_v', 'pop_v', 'spikes', 'watch_v']
PRE = '9658283e34b1c846b07c7631fc7853b4f06f7ddfbe0e38d329f9c243f99f946e'
PAR = '135220ca7e9bc01c5d88f4fee321dea7e2c5fe57c0c1db24de76187f5ee4962f'
d = Path(sys.argv[1])
ARMS = {f'v6_leak40_{l}': [f'v6_leak40_{l}_A', f'v6_leak40_{l}_B'] for l in ('axial-v1', 'malecns-hex-v2')}
ARMS.update({f'v5_gpu_{l}': [f'v5_gpu_{l}'] for l in ('axial-v1', 'malecns-hex-v2')})
errs, ref_shapes, ref_trace, ref_proto, ref_enc = [], {}, None, None, None


def pin(m):
    e = dict(m['encoder']); e.pop('lattice', None)
    return dict(protocol=m['protocol'], encoder=e)


for arm, parts in ARMS.items():
    lat = arm.split('_', 2)[-1]
    seen, metas = set(), []
    for pt in parts:
        try:
            m = json.load(open(d / f'{pt}.meta.json')); z = np.load(d / f'{pt}.npz')
        except FileNotFoundError as ex:
            errs.append(f'{pt}: missing {ex.filename}'); continue
        metas.append(m)
        if m.get('prereg_sha256') != PRE: errs.append(f'{pt}: prereg sha')
        if arm.startswith('v6') and m.get('params_sha256') != PAR: errs.append(f'{pt}: params sha')
        if m.get('lattice') != lat or m['encoder'].get('lattice', lat) != lat: errs.append(f'{pt}: lattice')
        p = pin(m)
        if ref_proto is None: ref_proto = p
        elif p != ref_proto: errs.append(f'{pt}: protocol/encoder differ from the first arm')
        t = z['trace_idx']
        if ref_trace is None: ref_trace = t
        elif not np.array_equal(t, ref_trace): errs.append(f'{pt}: trace_idx differs')
        for c in ['gray'] + [c for c in COND if c in m['conditions_done']]:
            for f in FIELDS:
                k = f'{c}__{f}'
                if k not in z.files: errs.append(f'{pt}: {k} absent'); continue
                a = z[k]
                if not np.isfinite(a).all(): errs.append(f'{pt}: {k} non-finite')
                if f in ref_shapes and f != 'pop_v' and f != 'watch_v' and a.shape != ref_shapes[f]:
                    errs.append(f'{pt}: {k} shape {a.shape} != {ref_shapes[f]}')
                if f in ('pop_v', 'watch_v') and c != 'gray':
                    key = f + '_cond'
                    if key in ref_shapes and a.shape != ref_shapes[key]: errs.append(f'{pt}: {k} shape {a.shape}')
                    ref_shapes.setdefault(key, a.shape)
                ref_shapes.setdefault(f, a.shape)
            if c != 'gray': seen.add(c)
        if len(parts) > 1 and 'gray__mean_v' in z.files:
            if pt == parts[0]: g0 = {f: z[f'gray__{f}'] for f in FIELDS}
            elif any(not np.array_equal(g0[f], z[f'gray__{f}']) for f in FIELDS): errs.append(f'{arm}: gray differs between parts')
    if len(metas) == len(parts) > 1:
        keys = ('prereg_sha256', 'params_sha256', 'lattice', 'yaw_axes', 'dynamics_pin', 'graded_report', 'io_map')
        for k in keys:
            if any(json.dumps(mm.get(k), sort_keys=True) != json.dumps(metas[0].get(k), sort_keys=True) for mm in metas):
                errs.append(f'{arm}: parts disagree on {k}')
    missing = [c for c in COND if c not in seen]
    if missing: errs.append(f'{arm}: missing conditions {missing}')
res = dict(evaluable=not errs, errors=errs, arms=list(ARMS))
(d / 's3_guard.json').write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
sys.exit(0 if not errs else 2)
