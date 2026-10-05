"""POST-LOCK DIAGNOSTIC (labelled; not preregistered; does NOT decide any gate).

Added 2026-10-05T15:11+02:00, after the v6a declaration was locked (632072b) and
BEFORE any v6a real-graph run or result existed.  Prompted by the coordinator's
data finding that the MaleCNS lamina is incomplete: 967/1,776 L1, 965/1,779 L2
and 990/1,772 L3 receive no R1-R6 synapse at all (verified independently in
this worktree; Nern et al. 2025 state the lamina is not covered and R1-R6 are
undercounted).  Population statistics that include input-less columns mix cells
that structurally cannot see the stimulus with cells that can.

This script splits every traced population into an INPUT-RECEIVING subset and
its complement, by anatomy only (never by response):

  stage 1  L1-L5 with > 0 synapses from a DRIVEN R1-R6 (the encoder's r_nodes);
  stage 2  Mi1, Tm3, Mi4, Mi9, Tm1, Tm2, Tm4, Tm9, CT1 with > 0 weight from a
           stage-1 cell;
  stage 3  T4a-d, T5a-d with > 0 weight from a stage-2 cell;

and reports the stage-gate statistic (per-cell F1 averaged over yaw CCW/CW)
as mean / median / p90 for each subset, for the given run and for the v4 / v5
references.  The declared stage gate (§10.8: right-eye p90 of the whole
population) is unchanged and is what the verdict rests on.
"""
import argparse
import json

import numpy as np

from brainlab.brain import _v3_policy_weight
from brainlab.graded_policy import load_cell_classes
from brainlab.photoreceptor_io import resolve_photoreceptor_io

p = argparse.ArgumentParser()
p.add_argument('tags', nargs='*', help='raw tags in docs/receipts/v6a_raw (stage phase done)')
p.add_argument('--out', default='docs/receipts/v6a_raw/posthoc_input_subset.json')
args = p.parse_args()

io = resolve_photoreceptor_io()
ct, _ = load_cell_classes()
n = len(ct)
with np.load('outputs/brainlab/malecns_v1/graph.npz') as g:
    arr = {k: g[k] for k in ('ptr', 'post', 'weight', 'ids')}
w = _v3_policy_weight(arr)
pre = np.repeat(np.arange(n), np.diff(arr['ptr']))
post = arr['post']


def receives_from(src_mask):
    sel = src_mask[pre]
    tot = np.bincount(post[sel], weights=np.abs(w[sel]).astype(np.float64), minlength=n)
    return tot > 0


driven = np.zeros(n, bool)
for eye in ('L', 'R'):
    driven[io.r_nodes[eye]] = True
LAM = ('L1', 'L2', 'L3', 'L4', 'L5')
MED = ('Mi1', 'Tm3', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm4', 'Tm9', 'CT1')
T45 = tuple(f'T{a}{s}' for a in '45' for s in 'abcd')
s1 = receives_from(driven) & np.isin(ct, LAM)
s2 = receives_from(s1) & np.isin(ct, MED)
s3 = receives_from(s2) & np.isin(ct, T45)
subset = s1 | s2 | s3
anatomy = {}
for t in LAM + MED + T45:
    m = ct == t
    anatomy[t] = dict(n=int(m.sum()), input_receiving=int((m & subset).sum()))
any_r = receives_from(ct == 'R1-R6')
anatomy['lamina_zero_R1R6_input'] = {t: dict(n=int((ct == t).sum()), zero=int(((ct == t) & ~any_r).sum()))
                                     for t in ('L1', 'L2', 'L3')}
POPS = ['L1', 'L2', 'L3', 'Mi1', 'Tm3', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm4', 'Tm9', 'T4', 'T5']


def members(k, eye):
    if k in ('T4', 'T5'):
        return np.concatenate([io.trace[f'{k}{s}_{eye}'] for s in 'abcd'])
    return np.asarray(io.trace[f'{k}_{eye}'])


def stats(path):
    out = {}
    with np.load(path) as raw:
        pos = {int(x): i for i, x in enumerate(raw['trace_idx'])}
        for eye in 'LR':
            for k in POPS:
                cells = members(k, eye)
                idx = np.array([pos[int(i)] for i in cells])
                A = np.mean([np.hypot(raw[f'{c}__f1_re'][idx], raw[f'{c}__f1_im'][idx])
                             for c in ('yaw_ccw', 'yaw_cw')], 0)
                inn = subset[cells]
                d = {}
                for lab, m in (('input_receiving', inn), ('input_less', ~inn), ('all', np.ones_like(inn))):
                    a = A[m]
                    d[lab] = (dict(n=int(m.sum()), mean=float(a.mean()), median=float(np.median(a)),
                                   p90=float(np.percentile(a, 90))) if m.any() else dict(n=0))
                out[f'{k}_{eye}'] = d
    return out


res = dict(label='POST-LOCK DIAGNOSTIC, not preregistered, does not decide any gate',
           added='2026-10-05T15:11+02:00, before any v6a real-graph result existed',
           prompted_by='incomplete MaleCNS lamina (coordinator data finding, verified here)',
           definition=__doc__, anatomy=anatomy, runs={})
for tag in ('v4_protocol9', 'v5_primary', 'v5_upper_S2'):
    res['runs'][tag] = stats(f'docs/receipts/v5_raw/{tag}.npz')
for tag in args.tags:
    res['runs'][tag] = stats(f'docs/receipts/v6a_raw/{tag}.npz')
json.dump(res, open(args.out, 'w'), indent=1)
print(json.dumps(anatomy, indent=0))
for k in POPS:
    print(f'{k:4s}', ' | '.join(
        f"{t[:9]} in {res['runs'][t][k + '_R']['input_receiving'].get('median', float('nan')):.3f}"
        f"/{res['runs'][t][k + '_R']['input_receiving'].get('p90', float('nan')):.3f}"
        f" less {res['runs'][t][k + '_R']['input_less'].get('median', float('nan')):.3f}"
        for t in res['runs']))
