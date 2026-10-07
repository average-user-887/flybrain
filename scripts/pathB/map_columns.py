"""Path B, phase 1: retinotopic paths covered end to end in MaleCNS (no simulation).

Read-only.  For every eye and every medulla column, intersects the real chain

    R1-R6 -> {L1, L2, L3} -> T4 inputs {Mi1, Tm3, Mi4, Mi9} -> T4a-d
                          -> T5 inputs {Tm1, Tm2, Tm4, Tm9} -> T5a-d

cell by cell, so a column counts as covered only when one connected chain of
real synapses runs from a photoreceptor to a T4/T5 cell of that column.  This is
a per-path intersection, not a product of per-cell coverage fractions.

Definitions (declared here, before any simulation):

* Synapse counts are the prepared graph's ``normalized/edges.arrow``
  ``synapse_count`` (the same edges the engine runs on).  Primary threshold
  THETA = 1 synapse; robustness threshold 5.
* lit L      : an L1/L2/L3 cell with >= THETA synapses from R1-R6 of the same eye.
* lit X      : a medulla input cell (Mi1, Tm3, Mi4, Mi9, Tm1, Tm2, Tm4, Tm9) with
               >= THETA synapses from lit L1/L2/L3 cells (strict chain, PRIMARY).
               SECONDARY variant ``_L5relay``: synapses from lit L5 also count,
               where L5 is lit by >= THETA synapses from lit L1/L2/L3 (Mi4's
               largest lamina input is L5, which has no photoreceptor input).
* T4/T5 cell : input-complete if it has >= THETA synapses from a lit cell of each
               of its four input types; partial if 1-3 types; blind if 0.
* column     : (eye, assignedOlHex1, assignedOlHex2).  Cells without a released
               hex (T4, T5, Tm3, R1-R6, one eye's L3 and Tm4) get the
               synapse-weighted majority hex of declared hex-annotated partners
               (DERIVED_FROM below).  A cell with no such partner has no column.
* column T4 arm: COMPLETE if >= 1 T4 cell of the column is input-complete; BLIND
               if no T4 cell of the column has any lit input; PARTIAL otherwise
               (also PARTIAL-NO-T4 never arises: columns without a T4 cell are
               reported as NO_CELL).  Same for T5.  Column overall: COMPLETE if
               both arms complete, BLIND if both arms blind/no-cell, else PARTIAL.

Output: a JSON receipt with the per-column table and per-stage synapse/sign
statistics, plus a CSV of the column table.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from brainlab.graph_identity import resolve_connectome_dir, resolve_graph_dir  # noqa: E402

LAMINA = ('L1', 'L2', 'L3')
T4_IN = ('Mi1', 'Tm3', 'Mi4', 'Mi9')
T5_IN = ('Tm1', 'Tm2', 'Tm4', 'Tm9')
T4 = ('T4a', 'T4b', 'T4c', 'T4d')
T5 = ('T5a', 'T5b', 'T5c', 'T5d')
MEDULLA = T4_IN + T5_IN
STUDY = ('R1-R6',) + LAMINA + ('L4', 'L5') + MEDULLA + T4 + T5
# Hex-less cells take the majority hex of these partners (direction, types).
DERIVED_FROM = {
    'T4': ('in', ('Mi1', 'Mi4', 'Mi9')),
    'T5': ('in', ('Tm1', 'Tm2', 'Tm9')),
    'Tm3': ('in', ('L1', 'Mi1')),
    'Tm4': ('in', ('L2', 'L5')),
    'L3': ('out', ('Mi9', 'Tm9', 'Mi1', 'Tm20')),
    'L1': ('out', ('Mi1', 'C3', 'L5')),
    'L2': ('out', ('Tm1', 'Tm2', 'L5')),
    'Mi1': ('in', ('L1', 'L5')),
    'Mi4': ('in', ('L5', 'Mi1')),
    'Mi9': ('in', ('L3',)),
    'Tm1': ('in', ('L2',)),
    'Tm2': ('in', ('L2',)),
    'Tm9': ('in', ('L3',)),
    'R1-R6': ('out', ('L1', 'L2', 'L3')),
}
SIGN = {'acetylcholine': +1, 'gaba': -1, 'glutamate': -1, 'histamine': -1}


def load(cdir: Path):
    import pyarrow.feather as feather
    nodes = feather.read_table(cdir / 'normalized/neurons.feather',
                               columns=['node_index', 'source_id', 'cell_type',
                                        'neurotransmitter']).to_pandas()
    ann = feather.read_table(cdir / 'annotations.feather',
                             columns=['bodyId', 'somaSide', 'rootSide', 'instance',
                                      'assignedOlHex1', 'assignedOlHex2']).to_pandas()
    nodes = nodes.join(ann.drop_duplicates('bodyId').set_index('bodyId'), on='source_id')
    assert (nodes.node_index.to_numpy() == np.arange(len(nodes))).all()
    edges = feather.read_table(cdir / 'normalized/edges.arrow').to_pandas()
    return nodes, edges


def side_of(nodes):
    soma = nodes.somaSide.fillna('').astype(str).to_numpy()
    root = nodes.rootSide.fillna('').astype(str).to_numpy()
    inst = nodes.instance.fillna('').astype(str).to_numpy()
    side = np.where(soma != '', soma, root)
    ctype = nodes.cell_type.fillna('').astype(str).to_numpy()
    r = ctype == 'R1-R6'
    suffix = np.array([s.rsplit('_', 1)[-1] for s in inst], dtype=object)
    side = np.where(r & np.isin(suffix, ['L', 'R']), suffix, side)
    return side.astype(str)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--theta', type=int, default=1)
    ap.add_argument('--theta-robust', type=int, default=5)
    args = ap.parse_args(argv)
    cdir, _ = resolve_connectome_dir(None)
    gdir, _ = resolve_graph_dir(None)
    nodes, edges = load(cdir)
    n = len(nodes)
    ctype = nodes.cell_type.fillna('').astype(str).to_numpy()
    side = side_of(nodes)
    nt = nodes.neurotransmitter.fillna('').astype(str).str.lower().to_numpy()
    h1 = nodes.assignedOlHex1.to_numpy(dtype=float)
    h2 = nodes.assignedOlHex2.to_numpy(dtype=float)

    # graph.npz weight signs, cross-check against transmitter sign
    with np.load(gdir / 'graph.npz', allow_pickle=False) as g:
        ptr, post, weight = g['ptr'], g['post'], g['weight']
    study = np.isin(ctype, STUDY)
    pre = edges.pre_index.to_numpy(np.int64)
    pst = edges.post_index.to_numpy(np.int64)
    syn = edges.synapse_count.to_numpy(np.int64)
    keep = study[pre] & study[pst]
    pre, pst, syn = pre[keep], pst[keep], syn[keep]
    same_eye = side[pre] == side[pst]

    # graph weight for each kept edge (by CSR lookup)
    src = np.repeat(np.arange(n, dtype=np.int64), np.diff(ptr))
    gkey = src * n + post.astype(np.int64)
    order = np.argsort(gkey)
    gkey_sorted = gkey[order]
    want = pre * n + pst
    pos = np.clip(np.searchsorted(gkey_sorted, want), 0, len(gkey_sorted) - 1)
    found = gkey_sorted[pos] == want
    gw = np.where(found, weight[order[pos]], np.nan).astype(np.float32)
    del src, gkey, order, gkey_sorted

    # ---- column assignment
    hexkey = np.full(n, None, dtype=object)
    native = ~np.isnan(h1) & ~np.isnan(h2)
    for i in np.flatnonzero(native & study):
        hexkey[i] = (int(h1[i]), int(h2[i]))
    hex_source = np.full(n, '', dtype=object)
    hex_source[native & study] = 'released'
    by_post = collections.defaultdict(list)
    by_pre = collections.defaultdict(list)
    for k in range(len(pre)):
        if same_eye[k]:
            by_post[pst[k]].append(k)
            by_pre[pre[k]].append(k)
    for i in np.flatnonzero(study & ~native):
        fam = 'T4' if ctype[i] in T4 else 'T5' if ctype[i] in T5 else ctype[i]
        if fam not in DERIVED_FROM:
            continue
        direction, partners = DERIVED_FROM[fam]
        votes = collections.Counter()
        for k in (by_post[i] if direction == 'in' else by_pre[i]):
            j = pre[k] if direction == 'in' else pst[k]
            if ctype[j] in partners and native[j]:
                votes[(int(h1[j]), int(h2[j]))] += int(syn[k])
        if votes:
            hexkey[i] = votes.most_common(1)[0][0]
            hex_source[i] = 'derived:' + direction + ':' + '+'.join(partners)

    results = {}
    for theta in (args.theta, args.theta_robust):
        for relay in (False, True):
            key = f'{theta}' + ('_L5relay' if relay else '')
            results[key] = analyse(theta, n, ctype, side, hexkey, pre, pst, syn, same_eye,
                                   relay_l5=relay)

    # ---- per-stage edge statistics (same eye, theta 1)
    stages = []
    pairs = [('R1-R6', t) for t in LAMINA] + \
            [(l, x) for l in LAMINA for x in MEDULLA] + [('L5', x) for x in MEDULLA] + \
            [(x, t) for x in T4_IN for t in T4] + [(x, t) for x in T5_IN for t in T5]
    for a, b in pairs:
        m = same_eye & (ctype[pre] == a) & (ctype[pst] == b)
        posts = np.flatnonzero(ctype == b)
        per_post = collections.Counter()
        for k in np.flatnonzero(m):
            per_post[pst[k]] += int(syn[k])
        vals = np.array([per_post.get(j, 0) for j in posts])
        gws = gw[m]
        stages.append(dict(pre=a, post=b, edges=int(m.sum()), synapses=int(syn[m].sum()),
                           post_cells=int(len(posts)),
                           post_cells_with_input=int((vals > 0).sum()),
                           median_syn_per_post_where_present=float(np.median(vals[vals > 0])) if (vals > 0).any() else 0.0,
                           transmitter=str(collections.Counter(nt[np.flatnonzero(ctype == a)]).most_common(1)[0][0]),
                           sign_by_transmitter=SIGN.get(collections.Counter(nt[np.flatnonzero(ctype == a)]).most_common(1)[0][0], 0),
                           graph_weight_sign=('mixed' if (gws > 0).any() and (gws < 0).any() else
                                              '+' if (gws > 0).all() else '-' if (gws < 0).all() else 'none')))

    # input composition of each medulla type (top 6 presynaptic types, same eye)
    composition = {}
    for x in MEDULLA:
        m = same_eye & (ctype[pst] == x)
        c = collections.Counter()
        for k in np.flatnonzero(m):
            c[ctype[pre[k]]] += int(syn[k])
        tot = sum(c.values())
        composition[x] = [(t, v, round(v / tot, 3)) for t, v in c.most_common(8)] if tot else []
    # composition restricted to the study set only; note it
    hex_src = collections.Counter((ctype[i] if ctype[i] not in T4 + T5 else ctype[i][:2], hex_source[i])
                                  for i in np.flatnonzero(study))
    out = dict(
        what='Path B phase 1: per-path intersection of R1-R6 -> L1/L2/L3 -> T4/T5 inputs -> T4/T5, MaleCNS',
        definitions=__doc__,
        data=dict(edges_file='normalized/edges.arrow', neurons_file='normalized/neurons.feather',
                  annotations_file='annotations.feather', graph='graph.npz',
                  edges_sha256=sha256(cdir / 'normalized/edges.arrow'),
                  graph_sha256=sha256(gdir / 'graph.npz')),
        column_assignment={f'{a}|{b}': v for (a, b), v in sorted(hex_src.items())},
        cross_eye_edges_excluded=int((~same_eye).sum()),
        stages=stages,
        medulla_input_composition_within_study_types=composition,
        results={str(k): {kk: vv for kk, vv in v.items() if kk not in ('columns', 'arrays')} for k, v in results.items()},
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'column_coverage.json').write_text(json.dumps(out, indent=1, default=str))
    for theta, res in results.items():
        with open(args.out / f'column_table_{theta}.csv', 'w', newline='') as fh:
            cols = list(res['columns'][0].keys())
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for row in res['columns']:
                w.writerow(row)
    # per-cell column assignment for phase 3 (study cells only)
    cells = np.flatnonzero(study & np.array([h is not None for h in hexkey]))
    colstate = {(r['eye'], r['hex1'], r['hex2']): r['overall'] for r in results[str(args.theta)]['columns']}
    np.savez_compressed(args.out / 'cell_columns.npz', node=cells,
                        cell_type=ctype[cells].astype('U8'), eye=side[cells].astype('U1'),
                        hex1=np.array([hexkey[i][0] for i in cells]), hex2=np.array([hexkey[i][1] for i in cells]),
                        column_state=np.array([colstate.get((side[i], hexkey[i][0], hexkey[i][1]), 'NONE') for i in cells]).astype('U8'))
    # hex adjacency evidence: offsets of L1 -> Tm3 synapses (Tm3 spans several columns)
    off = collections.Counter()
    for k in np.flatnonzero(same_eye & (ctype[pre] == 'L1') & (ctype[pst] == 'Mi4')):
        a, b = hexkey[pre[k]], hexkey[pst[k]]
        if a is not None and b is not None:
            off[(a[0] - b[0], a[1] - b[1])] += int(syn[k])
    print('L1->Mi4 hex offsets', off.most_common(10))
    off = collections.Counter()
    for k in np.flatnonzero(same_eye & (ctype[pre] == 'Mi4') & np.isin(ctype[pst], T4)):
        a, b = hexkey[pre[k]], hexkey[pst[k]]
        if a is not None and b is not None:
            off[(a[0] - b[0], a[1] - b[1])] += int(syn[k])
    print('Mi4->T4 hex offsets', off.most_common(10))
    # node lists for phase 3
    np.savez_compressed(args.out / 'column_sets.npz', **results[str(args.theta)]['arrays'])
    print(json.dumps(out['results'], indent=1, default=str)[:6000])


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def analyse(theta, n, ctype, side, hexkey, pre, pst, syn, same_eye, relay_l5=False):
    def syn_from(src_mask, post_types):
        """per post cell: synapses from presynaptic cells in src_mask (same eye)."""
        m = same_eye & src_mask[pre] & np.isin(ctype[pst], post_types)
        acc = np.zeros(n, dtype=np.int64)
        np.add.at(acc, pst[m], syn[m])
        return acc

    is_r = ctype == 'R1-R6'
    r_to_l = syn_from(is_r, LAMINA)
    lit_l = np.isin(ctype, LAMINA) & (r_to_l >= theta)
    l_to_x = syn_from(lit_l, MEDULLA)
    if relay_l5:
        # SECONDARY variant: L5 (no photoreceptor input of its own) as a relay,
        # lit when it receives >= theta synapses from lit L1/L2/L3.
        lit_l5 = (ctype == 'L5') & (syn_from(lit_l, ('L5',)) >= theta)
        l_to_x = l_to_x + syn_from(lit_l5, MEDULLA)
    all_l_to_x = syn_from(np.isin(ctype, LAMINA), MEDULLA)
    lit_x = np.isin(ctype, MEDULLA) & (l_to_x >= theta)
    # per T4/T5 cell, synapses from lit cells of each input type
    tt_in = {}
    for x in MEDULLA:
        tt_in[x] = syn_from(lit_x & (ctype == x), T4 + T5)
    tt_any = {x: syn_from(ctype == x, T4 + T5) for x in MEDULLA}
    state = np.full(n, '', dtype=object)
    nlit_types = np.zeros(n, dtype=np.int64)
    for fam, inputs in (('T4', T4_IN), ('T5', T5_IN)):
        cells = np.flatnonzero(np.isin(ctype, T4 if fam == 'T4' else T5))
        k = sum((tt_in[x][cells] >= theta).astype(int) for x in inputs)
        nlit_types[cells] = k
        state[cells] = np.where(k == 4, 'complete', np.where(k == 0, 'blind', 'partial'))
    # anatomically present (any, lit or not) input types per T4/T5 cell
    nany = np.zeros(n, dtype=np.int64)
    for fam, inputs in (('T4', T4_IN), ('T5', T5_IN)):
        cells = np.flatnonzero(np.isin(ctype, T4 if fam == 'T4' else T5))
        nany[cells] = sum((tt_any[x][cells] >= theta).astype(int) for x in inputs)

    cols = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in np.flatnonzero(np.isin(ctype, STUDY) & (side != '')):
        if hexkey[i] is None:
            continue
        fam = 'T4' if ctype[i] in T4 else 'T5' if ctype[i] in T5 else ctype[i]
        cols[(side[i], hexkey[i])][fam].append(i)

    rows = []
    summary = collections.Counter()
    arrays = collections.defaultdict(list)
    for (eye, hx), fams in sorted(cols.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        row = dict(eye=eye, hex1=hx[0], hex2=hx[1])
        for l in LAMINA:
            cells = fams.get(l, [])
            row[f'{l}_cells'] = len(cells)
            row[f'R_to_{l}_syn'] = int(r_to_l[cells].sum()) if cells else 0
            row[f'{l}_lit'] = int(lit_l[cells].any()) if cells else 0
        for x in MEDULLA:
            cells = fams.get(x, [])
            row[f'{x}_cells'] = len(cells)
            row[f'litL_to_{x}_syn'] = int(l_to_x[cells].sum()) if cells else 0
            row[f'{x}_lit'] = int(lit_x[cells].any()) if cells else 0
        arm_state = {}
        for fam, inputs in (('T4', T4_IN), ('T5', T5_IN)):
            cells = fams.get(fam, [])
            row[f'{fam}_cells'] = len(cells)
            for x in inputs:
                row[f'lit{x}_to_{fam}_syn'] = int(tt_in[x][cells].sum()) if cells else 0
            st = [state[c] for c in cells]
            row[f'{fam}_cells_complete'] = st.count('complete')
            row[f'{fam}_cells_partial'] = st.count('partial')
            row[f'{fam}_cells_blind'] = st.count('blind')
            row[f'{fam}_cells_anatomy_all4'] = int(sum(nany[c] == 4 for c in cells))
            if not cells:
                s = 'NO_CELL'
            elif 'complete' in st:
                s = 'COMPLETE'
            elif all(x == 'blind' for x in st):
                s = 'BLIND'
            else:
                s = 'PARTIAL'
            arm_state[fam] = s
            row[f'{fam}_arm'] = s
            summary[(eye, fam, s)] += 1
        a4, a5 = arm_state['T4'], arm_state['T5']
        if a4 == 'COMPLETE' and a5 == 'COMPLETE':
            overall = 'COMPLETE'
        elif a4 in ('BLIND', 'NO_CELL') and a5 in ('BLIND', 'NO_CELL'):
            overall = 'BLIND'
        else:
            overall = 'PARTIAL'
        lam = row['L1_lit'] or row['L2_lit'] or row['L3_lit']
        row['lamina_any_lit'] = int(bool(lam))
        row['overall'] = overall
        summary[(eye, 'overall', overall)] += 1
        summary[(eye, 'lamina_any_lit', int(bool(lam)))] += 1
        rows.append(row)
        for fam in ('T4', 'T5'):
            for c in fams.get(fam, []):
                arrays[f'{eye}_{fam}_{arm_state[fam]}'].append(c)
        for t in LAMINA + MEDULLA:
            for c in fams.get(t, []):
                arrays[f'{eye}_{t}_col{overall}'].append(c)
        arrays[f'{eye}_colhex_{overall}'].append(hx)

    cell_level = {}
    for t in LAMINA:
        cells = np.flatnonzero(ctype == t)
        for eye in ('L', 'R'):
            c = cells[side[cells] == eye]
            cell_level[f'{t}_{eye}_no_R_input_frac'] = round(float((r_to_l[c] < theta).mean()), 4)
    for t in MEDULLA:
        cells = np.flatnonzero(ctype == t)
        for eye in ('L', 'R'):
            c = cells[side[cells] == eye]
            cell_level[f'{t}_{eye}_lit_frac'] = round(float(lit_x[c].mean()), 4)
            cell_level[f'{t}_{eye}_any_L123_input_frac'] = round(float((all_l_to_x[c] >= theta).mean()), 4)
    for fam, types in (('T4', T4), ('T5', T5)):
        cells = np.flatnonzero(np.isin(ctype, types))
        for eye in ('L', 'R'):
            c = cells[side[cells] == eye]
            cnt = collections.Counter(state[c])
            cell_level[f'{fam}_{eye}_cells'] = int(len(c))
            cell_level[f'{fam}_{eye}_complete'] = cnt['complete']
            cell_level[f'{fam}_{eye}_partial'] = cnt['partial']
            cell_level[f'{fam}_{eye}_blind'] = cnt['blind']
            cell_level[f'{fam}_{eye}_anatomy_all4_inputs'] = int((nany[c] == 4).sum())
            cell_level[f'{fam}_{eye}_no_column'] = int(sum(hexkey[i] is None for i in c))
    out_arrays = {k: np.asarray(v, dtype=np.int64) for k, v in arrays.items()}
    out_arrays['lit_L'] = np.flatnonzero(lit_l)
    out_arrays['lit_X'] = np.flatnonzero(lit_x)
    return dict(theta=theta,
                column_summary={f'{a}|{b}|{c}': v for (a, b, c), v in sorted(summary.items(), key=str)},
                cell_level=cell_level, columns=rows, arrays=out_arrays)


if __name__ == '__main__':
    main()
