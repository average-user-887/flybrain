#!/usr/bin/env python3
"""Path A, phase 1: read-only pathway mapping in the MaleCNS v1.0 graph.

No simulation.  Reads the pinned prepared graph (``NEUROFLY_GRAPH_DIR``) and the
released MaleCNS tables (``NEUROFLY_CONNECTOME_DIR``), resolves the cell sets of
the Path A circuits by DECLARED rules (cell type / released synonym / exit
nerve), and reports for each stage pair:

* direct chemical connectivity (synapse contacts, connected pairs, v3 sign of
  the presynaptic cell under the declared v3 transmitter policy);
* the shortest weighted path from the source set to every target cell, with
  edge cost 1 / synapse_count, both over all edges and over excitatory-only
  edges (an inhibitory edge cannot carry excitation forward in the LIF proxy).

Outputs ``qualification/pathA/mapping.json`` and ``mapping.md``.  Paths in the
outputs are relative; no private locations are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brainlab.transmitter_policy import apply_policy, normalise  # noqa: E402

SYN = 0.275  # prepared-graph weight per contact (brainlab/prepare.py)

# Declared cell-set rules.  Every rule is a released annotation, never a guess.
# 'type' = regex on the released MaleCNS ``type`` (cell_type) column.
SETS = {
    # A1 antennal grooming
    'JO-CE': dict(type=r'^JO-(CA1|CA2|CL|CM|ED1|ED2_a|ED2_b|ED2_c|EV1|EV2|EV3|EV4|EV5|EV6)$',
                  note='Johnston organ zone C and E types (MaleCNS names; FlyWire types identical by flywireType)'),
    'JO-F': dict(type=r'^JO-(FD1|FD2|FV)$', note='Johnston organ zone F types'),
    'aBN1': dict(type=r'^SAD093$', note='aBN1 = SAD093 by Virtual Fly Brain FBbt_00111645 (hemibrain/FlyWire type SAD093); MaleCNS type SAD093'),
    'aDN1': dict(type=r'^DNg62$', note='MaleCNS synonym "Hampel 2015: aDN1"'),
    'aDN2': dict(type=r'^DNge078$', note='MaleCNS synonym "Hampel 2015: aDN2" (mancType DNfl023)'),
    'ProLN-MN': dict(superclass='vnc_motor', exitNerve='ProLN',
                     note='front-leg motor neurons: superclass vnc_motor with exitNerve ProLN'),
    # A2 escape
    'LC4': dict(type=r'^LC4$', note='visual projection type LC4'),
    'LPLC2': dict(type=r'^LPLC2$', note='visual projection type LPLC2'),
    'GF': dict(type=r'^DNp01$', note='giant fibre = DNp01 (MaleCNS synonym "Kennedy and Broadie 2018: GF")'),
    'TTMn': dict(type=r'^TTMn$', note='tergotrochanteral motor neuron, exitNerve PDMNp'),
    # calibration
    'sugarGRN-cal': dict(type=r'^(LB3b|LB3c|LB3d|LB4b)$', rootSide='R',
                         note='calibration set: MaleCNS types matching the FlyWire v783 types of the 20/21 '
                              'Shiu 2024 sugar-GRN root IDs found (LB3c 9, LB3b 2, LB3d 5, LB4b 4); one side'),
    'MN9': dict(type=r'^MN9$', note='proboscis motor neuron MN9'),
}

STAGES = {
    'A1': [('JO-CE', 'aBN1'), ('JO-F', 'aBN1'), ('JO-CE', 'aDN1'), ('JO-CE', 'aDN2'),
           ('aBN1', 'aDN1'), ('aBN1', 'aDN2'), ('aDN1', 'ProLN-MN'), ('aDN2', 'ProLN-MN'),
           ('JO-CE', 'ProLN-MN'), ('JO-F', 'ProLN-MN')],
    'A2': [('LC4', 'GF'), ('LPLC2', 'GF'), ('GF', 'TTMn'), ('LC4', 'TTMn'), ('LPLC2', 'TTMn')],
    'CAL': [('sugarGRN-cal', 'MN9')],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def load_nodes(cdir: Path):
    import pyarrow.feather as feather
    n = feather.read_table(cdir / 'normalized/neurons.feather').to_pandas()
    a = feather.read_table(cdir / 'annotations.feather',
                           columns=['bodyId', 'flywireType', 'hemibrainType', 'synonyms', 'class',
                                    'somaSide', 'rootSide', 'exitNerve', 'entryNerve', 'mancType']
                           ).to_pandas().drop_duplicates('bodyId').set_index('bodyId')
    n = n.join(a, on='source_id')
    assert np.array_equal(n.node_index.to_numpy(), np.arange(len(n)))
    return n


def resolve(n, rule):
    m = np.ones(len(n), dtype=bool)
    if 'type' in rule:
        m &= n.cell_type.fillna('').str.match(rule['type']).to_numpy()
    for col in ('superclass', 'exitNerve', 'rootSide'):
        if col in rule:
            m &= (n[col].fillna('') == rule[col]).to_numpy()
    return np.flatnonzero(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'qualification/pathA'))
    args = ap.parse_args()
    gdir = Path(os.environ['NEUROFLY_GRAPH_DIR'])
    cdir = Path(os.environ['NEUROFLY_CONNECTOME_DIR'])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n = load_nodes(cdir)
    with np.load(gdir / 'graph.npz', allow_pickle=False) as z:
        ptr, post, w1, ids = z['ptr'], z['post'], z['weight'], z['ids']
    assert np.array_equal(ids, n.source_id.to_numpy().astype(np.int64))
    labels = normalise(n.neurotransmitter.fillna('missing'))
    w3, policy_report = apply_policy(ptr, post, w1, labels)
    counts = np.rint(np.abs(w1) / SYN).astype(np.int64)
    pre = np.repeat(np.arange(len(ptr) - 1, dtype=np.int64), np.diff(ptr))
    nn = len(ids)

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    live = w3 != 0
    cost = 1.0 / np.maximum(counts, 1)
    g_all = csr_matrix((cost[live], (pre[live], post[live])), shape=(nn, nn))
    exc = w3 > 0
    g_exc = csr_matrix((cost[exc], (pre[exc], post[exc])), shape=(nn, nn))

    ct = n.cell_type.fillna('').to_numpy()
    sign_of = np.zeros(nn, dtype=np.int8)
    first = ptr[:-1]
    has = np.diff(ptr) > 0
    sign_of[has] = np.sign(w3[first[has]]).astype(np.int8)

    def label(i):
        return f"{ct[i] or '?'}#{int(ids[i])}"

    sets = {}
    for name, rule in SETS.items():
        idx = resolve(n, rule)
        sub = n.iloc[idx]
        sets[name] = dict(
            rule={k: v for k, v in rule.items() if k != 'note'}, note=rule['note'], count=int(len(idx)),
            node_index=[int(i) for i in idx], body_ids=[int(b) for b in sub.source_id],
            types=sub.cell_type.value_counts().to_dict(),
            transmitter=sub.neurotransmitter.value_counts().to_dict(),
            v3_sign=dict(zip(*[x.tolist() for x in np.unique(sign_of[idx], return_counts=True)])) if len(idx) else {},
            quality=sub.quality.value_counts().to_dict(),
            flywireType=sub.flywireType.fillna('<none>').value_counts().to_dict(),
            synonyms=sorted(set(sub.synonyms.dropna())),
            side=sub.somaSide.fillna(sub.rootSide).fillna('?').value_counts().to_dict(),
            exists=bool(len(idx)))
    sets_np = {k: np.array(v['node_index'], dtype=np.int64) for k, v in sets.items()}

    def direct(a, b):
        A, B = sets_np[a], sets_np[b]
        am = np.zeros(nn, bool); am[A] = True
        bm = np.zeros(nn, bool); bm[B] = True
        e = am[pre] & bm[post]
        syn = int(counts[e].sum())
        pos = int(counts[e & (w3 > 0)].sum()); neg = int(counts[e & (w3 < 0)].sum())
        zero = int(counts[e & (w3 == 0)].sum())
        return dict(synapses=syn, excitatory_synapses=pos, inhibitory_synapses=neg, zero_weight_synapses=zero,
                    connected_pairs=int(e.sum()),
                    targets_contacted=int(len(np.unique(post[e]))), targets=int(len(B)))

    def paths(a, b, graph, tag):
        A, B = sets_np[a], sets_np[b]
        if not len(A) or not len(B):
            return []
        dist, pred, src = dijkstra(graph, directed=True, indices=A, return_predecessors=True, min_only=True)
        rows = []
        for j in B:
            if not np.isfinite(dist[j]):
                rows.append(dict(target=label(j), reachable=False))
                continue
            hops = [int(j)]
            while hops[-1] not in set(A.tolist()) and pred[hops[-1]] >= 0:
                hops.append(int(pred[hops[-1]]))
            hops = hops[::-1]
            steps = []
            for u, v in zip(hops[:-1], hops[1:]):
                e = ptr[u] + np.flatnonzero(post[ptr[u]:ptr[u + 1]] == v)
                steps.append(dict(pre=label(u), post=label(v), synapses=int(counts[e].sum()),
                                  sign=int(np.sign(w3[e[0]]))))
            rows.append(dict(target=label(j), reachable=True, cost=float(dist[j]), hops=len(hops) - 1,
                             path=[label(h) for h in hops], steps=steps,
                             min_synapses=min(s['synapses'] for s in steps) if steps else None))
        rows.sort(key=lambda r: (not r['reachable'], r.get('cost', 1e9)))
        return rows

    stages = {}
    for test, pairs in STAGES.items():
        stages[test] = []
        for a, b in pairs:
            stages[test].append(dict(source=a, target=b, direct=direct(a, b),
                                     shortest_all=paths(a, b, g_all, 'all')[:6],
                                     shortest_excitatory=paths(a, b, g_exc, 'exc')[:6]))

    # Shiu 2024 mechanism probe: inhibitory cells postsynaptic to JO-F and presynaptic to aBN1.
    def bridge(a, b, sign):
        A, B = sets_np[a], sets_np[b]
        am = np.zeros(nn, bool); am[A] = True
        bm = np.zeros(nn, bool); bm[B] = True
        e1 = am[pre]
        mids_in = {}
        for u, v, c in zip(pre[e1], post[e1], counts[e1]):
            mids_in[int(v)] = mids_in.get(int(v), 0) + int(c)
        e2 = bm[post]
        out = []
        for u, v, c, ww in zip(pre[e2], post[e2], counts[e2], w3[e2]):
            if int(u) in mids_in and np.sign(ww) == sign:
                out.append(dict(mid=label(u), in_from_source=mids_in[int(u)], out_to_target=int(c),
                                transmitter=str(n.neurotransmitter.iat[int(u)])))
        out.sort(key=lambda r: -min(r['in_from_source'], r['out_to_target']))
        return out[:15]

    probes = dict(
        JOF_inhibitory_onto_aBN1=bridge('JO-F', 'aBN1', -1),
        JOCE_inhibitory_onto_aBN1=bridge('JO-CE', 'aBN1', -1),
        JOCE_excitatory_onto_aDN1=bridge('JO-CE', 'aDN1', 1),
        LC_excitatory_onto_GF=bridge('LPLC2', 'GF', 1),
    )
    # Every input to TTMn and every output of GF, chemical only.
    def top_inputs(b, k=12):
        B = sets_np[b]
        bm = np.zeros(nn, bool); bm[B] = True
        e = bm[post]
        agg = {}
        for u, c, ww in zip(pre[e], counts[e], w3[e]):
            key = (ct[u] or '?', int(np.sign(ww)))
            agg[key] = agg.get(key, 0) + int(c)
        rows = sorted(agg.items(), key=lambda kv: -kv[1])[:k]
        return [dict(type=t, sign=s, synapses=c) for (t, s), c in rows]

    def top_outputs(a, k=12):
        A = sets_np[a]
        am = np.zeros(nn, bool); am[A] = True
        e = am[pre]
        agg = {}
        for v, c in zip(post[e], counts[e]):
            agg[ct[v] or '?'] = agg.get(ct[v] or '?', 0) + int(c)
        return [dict(type=t, synapses=c) for t, c in sorted(agg.items(), key=lambda kv: -kv[1])[:k]]

    inventories = {f'inputs_{k}': top_inputs(k) for k in ('aBN1', 'aDN1', 'aDN2', 'GF', 'TTMn', 'MN9')}
    inventories.update({f'outputs_{k}': top_outputs(k) for k in ('aBN1', 'aDN1', 'aDN2', 'GF')})

    result = dict(
        schema='neurofly.pathA.mapping/1', dataset='MaleCNS v1.0',
        graph_npz_sha256=sha256(gdir / 'graph.npz'),
        neurons_feather_sha256=sha256(cdir / 'normalized/neurons.feather'),
        annotations_feather_sha256=sha256(cdir / 'annotations.feather'),
        v3_policy=dict((k, v) for k, v in policy_report.items() if isinstance(v, (str, int, float, bool))),
        edge_cost='1/synapse_count; zero-weight (modulatory) edges excluded; excitatory-only variant uses v3 weight > 0',
        electrical_synapses='NOT IN DATA: the MaleCNS release has chemical synapses only',
        sets=sets, stages=stages, probes=probes, inventories=inventories)
    (out / 'mapping.json').write_text(json.dumps(result, indent=1, default=int))
    print(json.dumps({k: dict(count=v['count'], types=v['types'], sign=v['v3_sign']) for k, v in sets.items()}, indent=0, default=int))
    for test, rows in stages.items():
        for r in rows:
            d = r['direct']
            best = next((p for p in r['shortest_excitatory'] if p['reachable']), None)
            print(test, r['source'], '->', r['target'], 'direct', d['synapses'], f"(+{d['excitatory_synapses']}/-{d['inhibitory_synapses']})",
                  'pairs', d['connected_pairs'], 'tgt', f"{d['targets_contacted']}/{d['targets']}",
                  '| best exc path', best and ' > '.join(best['path']), best and best['min_synapses'])
    print(json.dumps(probes, indent=0)[:4000])
    print(json.dumps(inventories, indent=0)[:6000])


if __name__ == '__main__':
    main()
