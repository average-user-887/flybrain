#!/usr/bin/env python3
"""Path A, read-only: inventory of the existing PVLP151 -> GF edges and of the 2-hop
PVLP151 -> X -> GF routes (and the GF -> PVLP151 feedback), for the proposed
direct-edge-deletion counterfactual.  No dynamics, no graph write.

Weights: ``graph_weight`` is the pinned graph.npz value, ``contacts`` = |graph_weight| /
synaptic_scale (0.275 mV per contact), ``v3_weight`` the value after the engine's v3
transmitter policy (the weight a run uses), cross-checked against edges.arrow synapse_count.
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
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import pathA_run as run  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'qualification/pathA/pvlp151_gf_inventory.json'))
    args = ap.parse_args()
    contract = json.loads((ROOT / 'qualification/pathA/contract.json').read_text())
    prereg = json.loads((ROOT / 'qualification/pathA/pvlp151_cf_prereg.json').read_text())
    gdir, cdir = Path(os.environ['NEUROFLY_GRAPH_DIR']), Path(os.environ['NEUROFLY_CONNECTOME_DIR'])
    gsha = sha(gdir / 'graph.npz')
    assert gsha == contract['data']['graph_npz_sha256']
    scale = contract['protocol']['synaptic_scale_mV_per_contact']
    with np.load(gdir / 'graph.npz', allow_pickle=False) as z:
        ptr, post, w_raw, ids = z['ptr'], z['post'], z['weight'], z['ids']
    w3 = run.load_v3_arrays(gdir, cdir, contract['data']['neurons_feather_sha256'])['weight']
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc
    n = feather.read_table(cdir / 'normalized/neurons.feather',
                           columns=['node_index', 'cell_type', 'neurotransmitter']).to_pandas()
    ct, nt = n.cell_type.fillna('').to_numpy(), n.neurotransmitter.fillna('?').to_numpy()
    t = ipc.open_file(pa.memory_map(str(cdir / 'normalized/edges.arrow'))).read_all()
    epre, epost, esc = (t[c].to_numpy() for c in ('pre_index', 'post_index', 'synapse_count'))
    pv = np.array(prereg['sets']['PVLP151']['node_index'])
    gf = np.array(contract['sets']['GF']['node_index'])
    pre = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))

    def edge_rows(src, dst):
        e = np.flatnonzero(np.isin(pre, src) & np.isin(post, dst))
        out = []
        for i in e:
            a, b = int(pre[i]), int(post[i])
            m = (epre == a) & (epost == b)
            out.append(dict(edge_index=int(i), pre_node=a, pre_body=int(ids[a]), pre_type=ct[a], pre_transmitter=nt[a],
                            post_node=b, post_body=int(ids[b]), post_type=ct[b],
                            graph_weight=float(w_raw[i]), contacts=int(round(abs(float(w_raw[i])) / scale)),
                            edges_arrow_synapse_count=int(esc[m].sum()), v3_weight=float(w3[i]),
                            v3_sign='+' if w3[i] > 0 else ('-' if w3[i] < 0 else '0')))
        return out

    direct = edge_rows(pv, gf)
    feedback = edge_rows(gf, pv)
    # 2-hop PVLP151 -> X -> GF (X not PVLP151/GF): per intermediate cell type, contacts and v3 weight sums.
    out_e = np.flatnonzero(np.isin(pre, pv))
    mids = np.setdiff1d(np.unique(post[out_e]), np.concatenate([pv, gf]))
    in_gf = np.flatnonzero(np.isin(post, gf) & np.isin(pre, mids))
    first = {}
    for i in out_e:
        b = int(post[i])
        if b in set(mids.tolist()):
            first.setdefault(b, [0, 0.0])
            first[b][0] += int(round(abs(float(w_raw[i])) / scale)); first[b][1] += float(w3[i])
    second = {}
    for i in in_gf:
        a = int(pre[i])
        second.setdefault(a, [0, 0.0])
        second[a][0] += int(round(abs(float(w_raw[i])) / scale)); second[a][1] += float(w3[i])
    by_type = {}
    for x in sorted(set(first) & set(second)):
        d = by_type.setdefault(ct[x] or f'(untyped {x})', dict(cells=0, transmitter=set(), pvlp151_to_x_contacts=0,
                                                            pvlp151_to_x_v3_weight=0.0, x_to_gf_contacts=0,
                                                            x_to_gf_v3_weight=0.0, min_leg_contacts_sum=0))
        d['cells'] += 1; d['transmitter'].add(nt[x])
        d['pvlp151_to_x_contacts'] += first[x][0]; d['pvlp151_to_x_v3_weight'] += first[x][1]
        d['x_to_gf_contacts'] += second[x][0]; d['x_to_gf_v3_weight'] += second[x][1]
        d['min_leg_contacts_sum'] += min(first[x][0], second[x][0])
    for d in by_type.values():
        d['transmitter'] = sorted(d['transmitter'])
        d['x_to_gf_sign'] = '+' if d['x_to_gf_v3_weight'] > 0 else ('-' if d['x_to_gf_v3_weight'] < 0 else '0')
    routes = sorted(by_type.items(), key=lambda kv: -kv[1]['min_leg_contacts_sum'])
    inv = dict(
        schema='neurofly.pathA.pvlp151_gf_inventory/1',
        note='READ-ONLY anatomical inventory. Counts and weights are wiring facts, not measured causal contributions.',
        graph_npz_sha256=gsha, neurons_feather_sha256=sha(cdir / 'normalized/neurons.feather'),
        edges_arrow_sha256=sha(cdir / 'normalized/edges.arrow'), synaptic_scale_mV_per_contact=scale,
        pvlp151=dict(node_index=pv.tolist(), body_ids=[int(ids[i]) for i in pv],
                     out_edges=int(len(out_e)), out_contacts=int(round(np.abs(w_raw[out_e]).sum() / scale))),
        gf=dict(node_index=gf.tolist(), body_ids=[int(ids[i]) for i in gf]),
        direct_pvlp151_to_gf=dict(edges=len(direct), contacts=sum(r['contacts'] for r in direct),
                                  v3_weight_sum=sum(r['v3_weight'] for r in direct),
                                  fraction_of_pvlp151_out_contacts=sum(r['contacts'] for r in direct)
                                  / max(1, int(round(np.abs(w_raw[out_e]).sum() / scale))), rows=direct),
        feedback_gf_to_pvlp151=dict(edges=len(feedback), contacts=sum(r['contacts'] for r in feedback), rows=feedback),
        two_hop_pvlp151_x_gf=dict(
            intermediate_cells=len(set(first) & set(second)), intermediate_types=len(routes),
            total_min_leg_contacts=sum(d['min_leg_contacts_sum'] for _, d in routes),
            rule='X = any neuron except PVLP151/GF receiving a PVLP151 edge and sending a GF edge; '
                 'min_leg_contacts_sum = sum over X cells of min(PVLP151->X, X->GF) contacts (an anatomical bound, not a flow)',
            by_type=[dict(type=k, **v) for k, v in routes]),
    )
    Path(args.out).write_text(json.dumps(inv, indent=1) + '\n')
    print(sha(args.out))


if __name__ == '__main__':
    main()
