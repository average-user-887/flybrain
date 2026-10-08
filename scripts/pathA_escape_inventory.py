#!/usr/bin/env python3
"""Read-only GF->TTMn/GFC2 static inventory; imports no simulation runner.

Two-hop contact totals describe anatomy, not transmission, route share or causality.
All data identities and selected graph/Arrow contact counts must agree before output.
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
SETS = ('GF', 'TTMn', 'GFC2', 'PSI', 'DLMn')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_pin(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'{Path(path).name}: pinned identity mismatch')
    return actual


def select_edges(arrays, sets):
    """Return the direct, feedback and every GF->X->TTMn leg, excluding endpoint X."""
    ptr, post = arrays['ptr'], arrays['post']
    pre = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    gf, ttm = sets['GF'], sets['TTMn']
    out = np.flatnonzero(np.isin(pre, gf))
    incoming = np.flatnonzero(np.isin(post, ttm))
    mids = np.intersect1d(post[out], pre[incoming])
    mids = np.setdiff1d(mids, np.concatenate([gf, ttm]))
    groups = {}
    for a, b in (('GF', 'TTMn'), ('GF', 'GFC2'), ('GFC2', 'TTMn'),
                 ('TTMn', 'GF'), ('GFC2', 'GF'), ('TTMn', 'GFC2'),
                 ('GF', 'PSI'), ('PSI', 'DLMn')):
        groups[f'{a}->{b}'] = np.flatnonzero(np.isin(pre, sets[a]) & np.isin(post, sets[b]))
    groups['GF->X'] = out[np.isin(post[out], mids)]
    groups['X->TTMn'] = incoming[np.isin(pre[incoming], mids)]
    return pre, mids, groups


def inventory(arrays, weights_v3, sets, labels, arrow_counts, scale):
    pre, mids, groups = select_edges(arrays, sets)
    ids, post, raw = arrays['ids'], arrays['post'], arrays['weight']
    selected = sorted(set(int(e) for group in groups.values() for e in group))
    rows = {}
    for e in selected:
        a, b = int(pre[e]), int(post[e])
        contact = arrow_counts.get((a, b))
        if contact is None or contact <= 0:
            raise ValueError(f'edge {e}: missing/invalid Arrow contacts')
        # Prepared weights are float32; compare to float32 contact conversion.
        expected = np.float32(contact * scale)
        if not np.isclose(abs(raw[e]), expected, rtol=2e-7, atol=1e-6):
            raise ValueError(f'edge {e}: graph/Arrow contact mismatch')
        rows[e] = dict(edge_index=e, pre_node=a, post_node=b,
                       pre_body=int(ids[a]), post_body=int(ids[b]), contacts=int(contact),
                       graph_weight=float(raw[e]), v3_weight=float(weights_v3[e]),
                       pre_type=labels[a]['cell_type'], pre_transmitter=labels[a]['neurotransmitter'],
                       post_type=labels[b]['cell_type'])

    def summary(edges):
        rr = [rows[int(e)] for e in edges]
        return dict(edge_count=len(rr), contacts=sum(r['contacts'] for r in rr),
                    positive_contacts=sum(r['contacts'] for r in rr if r['v3_weight'] > 0),
                    negative_contacts=sum(r['contacts'] for r in rr if r['v3_weight'] < 0),
                    zero_fast_contacts=sum(r['contacts'] for r in rr if r['v3_weight'] == 0), rows=rr)

    routes = []
    for x in mids:
        x = int(x)
        first = groups['GF->X'][post[groups['GF->X']] == x]
        second = groups['X->TTMn'][pre[groups['X->TTMn']] == x]
        a, b = summary(first), summary(second)
        routes.append(dict(node_index=x, body_id=int(ids[x]), **labels[x],
                           gf_to_x=a, x_to_ttmn=b))
    return dict(direct_and_feedback={k: summary(v) for k, v in groups.items() if 'X' not in k},
                two_hop_gf_x_ttmn=dict(intermediate_count=len(routes), rows=routes),
                note='Static existing chemical graph only. Separate leg contacts are not a flow, '
                     'pathway fraction, transmission measurement, necessity or mediation result.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--graph-dir', type=Path, default=os.environ.get('NEUROFLY_GRAPH_DIR'))
    ap.add_argument('--connectome-dir', type=Path, default=os.environ.get('NEUROFLY_CONNECTOME_DIR'))
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    if args.graph_dir is None or args.connectome_dir is None:
        ap.error('supply both data directories or their NEUROFLY environment variables')
    q = ROOT / 'qualification/pathA'
    contract = json.loads((q / 'contract.json').read_text())
    cf = json.loads((q / 'pvlp151_cf_prereg.json').read_text())
    accepted = json.loads((q / 'pvlp151_gf_inventory.json').read_text())
    files = {'graph_npz': args.graph_dir / 'graph.npz',
             'neurons_feather': args.connectome_dir / 'normalized/neurons.feather',
             'annotations_feather': args.connectome_dir / 'annotations.feather',
             'edges_arrow': args.connectome_dir / 'normalized/edges.arrow'}
    pins = {k + '_sha256': verify_pin(p, accepted['edges_arrow_sha256'] if k == 'edges_arrow'
                                    else contract['data'][k + '_sha256']) for k, p in files.items()}
    pins.update(contract_sha256=sha(q / 'contract.json'),
                accepted_cf_prereg_sha256=sha(q / 'pvlp151_cf_prereg.json'),
                accepted_direct_edge_prereg_sha256=sha(q / 'pvlp151_gf_edge_prereg.json'),
                accepted_source_sha='457d703038d6e7b8b77c71cd3ee933e00979a552',
                inventory_helper_sha256=sha(__file__))
    with np.load(files['graph_npz'], allow_pickle=False) as z:
        arrays = {k: z[k] for k in ('ptr', 'post', 'weight', 'ids')}
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc
    from brainlab.transmitter_policy import apply_policy
    table = feather.read_table(files['neurons_feather']).to_pandas().sort_values('node_index')
    if not np.array_equal(table.node_index.to_numpy(), np.arange(len(arrays['ids']))) or not np.array_equal(
            table.source_id.to_numpy(), arrays['ids']):
        raise ValueError('neuron node/body ordering differs from graph')
    labels = table[['cell_type', 'neurotransmitter', 'quality']].fillna('unknown').to_dict(orient='records')
    merged = dict(contract['sets'], **cf['sets'])
    sets = {k: np.asarray(merged[k]['node_index'], dtype=np.int64) for k in SETS}
    identities = {}
    ann = feather.read_table(files['annotations_feather']).to_pandas().set_index('bodyId')
    for k, nodes in sets.items():
        if arrays['ids'][nodes].tolist() != merged[k]['body_ids']:
            raise ValueError(f'{k}: frozen body identity mismatch')
        if any(labels[int(i)]['cell_type'] != ('DNp01' if k == 'GF' else k) for i in nodes if k != 'DLMn'):
            raise ValueError(f'{k}: frozen type mismatch')
        expected_type = 'DNp01' if k == 'GF' else k
        typed = np.flatnonzero(table.cell_type.str.startswith('DLMn').fillna(False).to_numpy()
                              if k == 'DLMn' else table.cell_type.eq(expected_type).to_numpy())
        if not np.array_equal(nodes, typed):
            raise ValueError(f'{k}: frozen set does not cover exactly the declared type')
        identities[k] = []
        for i in nodes:
            body = int(arrays['ids'][i])
            metadata = ann.loc[body]
            if not hasattr(metadata, 'dtype'):
                raise ValueError(f'{k}: duplicate annotation body')
            identities[k].append(dict(node_index=int(i), body_id=body, **labels[int(i)],
                                      root_side=None if metadata.get('rootSide') != metadata.get('rootSide')
                                      else str(metadata.get('rootSide', 'unknown')),
                                      annotation_type=str(metadata.get('type', 'unknown'))))
    v3, _ = apply_policy(arrays['ptr'], arrays['post'], arrays['weight'], table.neurotransmitter.to_numpy())
    pins['v3_weight_sha256'] = hashlib.sha256(np.ascontiguousarray(v3, dtype=np.float32).tobytes()).hexdigest()
    pre, _, groups = select_edges(arrays, sets)
    selected = np.unique(np.concatenate(list(groups.values())))
    n = len(arrays['ids'])
    wanted = pre[selected].astype(np.int64) * n + arrays['post'][selected]
    with pa.memory_map(str(files['edges_arrow'])) as mm:
        t = ipc.open_file(mm).read_all()
        a, b, counts = (t[c].to_numpy() for c in ('pre_index', 'post_index', 'synapse_count'))
        take = np.flatnonzero(np.isin(a.astype(np.int64) * n + b, wanted))
        ac = {}
        for j in take:
            pair = (int(a[j]), int(b[j]))
            ac[pair] = ac.get(pair, 0) + int(counts[j])
    result = inventory(arrays, v3, sets, labels, ac, contract['protocol']['synaptic_scale_mV_per_contact'])
    # Re-hash inputs before writing: even concurrent data changes fail closed.
    for k, p in files.items():
        verify_pin(p, pins[k + '_sha256'])
    out = dict(schema='neurofly.pathA.escape_inventory/1', pins=pins, identities=identities, **result)
    with args.out.open('x') as stream:
        stream.write(json.dumps(out, indent=1) + '\n')
    print(json.dumps(dict(output_sha256=sha(args.out),
                          direct_contacts={k: v['contacts'] for k, v in result['direct_and_feedback'].items()},
                          intermediate_count=result['two_hop_gf_x_ttmn']['intermediate_count'])))


if __name__ == '__main__':
    main()
