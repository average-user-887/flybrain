#!/usr/bin/env python3
"""Path A: write the preregistration for the PVLP151->GF direct-edge-deletion counterfactual
(no run).  Reads the frozen contract, the accepted PVLP151 prereg, the read-only inventory,
the two count manifests and the reset fixture, and computes the pinned v3 weight hashes from
the pinned graph (no dynamics)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import pathA_provenance as prov  # noqa: E402
import pathA_run as run  # noqa: E402

Q = ROOT / 'qualification/pathA'
BASES = ['A2_IRR_LC15_200', 'A2_LC4_200', 'A2_LPLC2_200', 'A2_LC4_0']
FIXTURE = 'fixtures/pvlp151_gf_reset_state_v3.npz'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    contract, cf = json.loads((Q / 'contract.json').read_text()), json.loads((Q / 'pvlp151_cf_prereg.json').read_text())
    inv = json.loads((Q / 'pvlp151_gf_inventory.json').read_text())
    edges = [dict(edge_index=e['edge_index'], pre_body=e['pre_body'], post_body=e['post_body'], pre_node=e['pre_node'],
                  post_node=e['post_node'], contacts=e['contacts'], v3_weight=e['v3_weight'])
             for e in inv['direct_pvlp151_to_gf']['rows']]
    listed = sorted(e['edge_index'] for e in edges)
    seeds = contract['protocol']['seeds']
    conds = []
    for b in BASES:
        conds += [dict(id=f'{b}_REPLAY', base_condition=b, arm='intact', counterfactual=None, add_silence=[],
                       zero_edges=[], seeds=seeds),
                  dict(id=f'{b}_DEL_PVLP151_GF', base_condition=b, arm='direct_edge_deletion',
                       counterfactual='delete_PVLP151_GF_edges', add_silence=[], zero_edges=listed, seeds=seeds),
                  dict(id=f'{b}_CF_silence_PVLP151', base_condition=b, arm='all_output_clamp',
                       counterfactual='silence_PVLP151', add_silence=['PVLP151'], zero_edges=[], seeds=seeds)]
        if b != 'A2_LC4_0':
            conds.append(dict(id=f'{b}_SHAM_DEL_EMPTY', base_condition=b, arm='sham_empty_deletion',
                              counterfactual='sham_empty_deletion', add_silence=[], zero_edges=[], weight_copy=True,
                              seeds=[0]))
    gdir, cdir = Path(os.environ['NEUROFLY_GRAPH_DIR']), Path(os.environ['NEUROFLY_CONNECTOME_DIR'])
    assert sha(gdir / 'graph.npz') == contract['data']['graph_npz_sha256']
    arrays = run.load_v3_arrays(gdir, cdir, contract['data']['neurons_feather_sha256'])
    tmp = dict(direct_edges=edges, direct_edges_from='PVLP151', direct_edges_to='GF', direct_edge_count=8)
    sets = {k: np.array(v['node_index']) for k, v in dict(contract['sets'], **cf['sets']).items()}
    assert run.check_direct_edges(arrays, sets, tmp).tolist() == listed
    base_w = hashlib.sha256(arrays['weight'].tobytes()).hexdigest()
    del_w = hashlib.sha256(run.deleted_weight(arrays, listed).tobytes()).hexdigest()
    with np.load(Q / FIXTURE, allow_pickle=False) as z:
        order = [k[len('state__'):] for k in z.files if k.startswith('state__')]
    import pathA_gf_edge_analyse as ga
    fdig, probs = ga.fixture_digest(Q / FIXTURE, order, 'v3', len(arrays['ids']))
    assert not probs, probs
    pins = dict(graph_npz_sha256=contract['data']['graph_npz_sha256'],
                neurons_feather_sha256=contract['data']['neurons_feather_sha256'],
                engine_sha256=prov.engine_sha256(),
                original_a2_code_sha='6bc971fbd52e6328367ffcc325462923fbf84662',
                accepted_clamp_code_sha='33110a4f3292ab3ce8a5c64ee8a17b2626e1cec1',
                accepted_clamp_prereg_sha256=sha(Q / 'pvlp151_cf_prereg.json'),
                frozen_counts_manifest='pvlp151_cf_original_counts.sha256.json',
                frozen_counts_manifest_sha256=sha(Q / 'pvlp151_cf_original_counts.sha256.json'),
                frozen_counts_subset_digest='82c13d3cbd3cec64270b6d5b0a546ff37ffa710936a5be1b11c876afe00de64c',
                clamp_counts_manifest='pvlp151_cf_clamp_counts.sha256.json',
                clamp_counts_manifest_sha256=sha(Q / 'pvlp151_cf_clamp_counts.sha256.json'),
                clamp_counts_digest=json.loads((Q / 'pvlp151_cf_clamp_counts.sha256.json').read_text())['sha_of_shas']['value'],
                inventory_sha256=sha(Q / 'pvlp151_gf_inventory.json'),
                reset_fixture_file=FIXTURE, reset_fixture_sha256=sha(Q / FIXTURE), reset_state_sha256=fdig,
                state_arrays_order=order, reset_fixture_note='one fixture: every arm resets to the same state '
                'shape (same n, same dynamics); written by pathA_run.py --save-reset-fixture without stepping',
                base_weight_sha256=base_w, deleted_weight_sha256=del_w)
    p = {
        'schema': 'neurofly.pathA.cf_prereg/3',
        'title': 'PVLP151->GF direct-edge deletion vs all-output clamp',
        'question': 'How do the paired changes of GF and the other readout populations under deletion of ONLY the 8 '
                    'existing PVLP151->GF edges compare with those under the accepted all-output clamp of PVLP151? '
                    'Descriptive comparison only.',
        'label': 'LABELLED OFFLINE COUNTERFACTUAL: existing wiring only, no edge added, not a fix, not an A2 re-grade',
        'status': 'FROZEN FOR ROOT LAUNCH DECISION. Not launched. Any change gives a new sha256.',
        'origin': 'ASTRA_PVLP151_FINAL_DECISION_20261008 items 8-9 (question raised after the accepted clamp result).',
        'frozen_contract_sha256': sha(Q / 'contract.json'),
        'not_a_fix': cf['not_a_fix'],
        'owner_rules': cf['owner_rules'],
        'pins': pins,
        'direct_edges_from': 'PVLP151', 'direct_edges_to': 'GF', 'direct_edge_count': 8, 'direct_edges': edges,
        'arms': {
            'intact': 'frozen A2 condition, unchanged (sham replay); base v3 weights',
            'direct_edge_deletion': 'the 8 listed PVLP151->GF edges (603 contacts) set to 0 in an in-memory copy of '
                                    'the v3 weights; PVLP151 not clamped; nothing else changes; graph file untouched',
            'all_output_clamp': 'the accepted intervention, unchanged (frozen silence_drive on PVLP151)',
            'sham_empty_deletion': 'seed 0: the weight-copy path with no edge; must equal the intact row bit for bit',
        },
        'sets': cf['sets'],
        'readouts': cf['readouts'],
        'conditions': conds,
        'gates': {
            'G1_clamp_zero': 'PVLP151 spike count 0 in every clamp row.',
            'G2_clean_state': 'the pinned reset fixture passes an independent check (v float32 at -52 mV everywhere, '
                              'every other state array and clock scalar zero, trailing size n) and its canonical '
                              'digest (name, dtype, shape, bytes, scalars) equals reset_state_sha256; every row records '
                              'initial_state_clean = true, state_sha256 = reset_state_sha256 (so no row inherits a '
                              'parent state), and an RNG seed sequence and state digest equal to the frozen input '
                              'definition [seed, round(10*rate), n_activated].',
            'G3_identical_inputs': 'all arms of one base condition and seed (and the seed-0 sham) carry the same '
                                   '64-hex input_sha256.',
            'G4_deletion_exact': 'the runner verified before any row that the 8 listed edges are exactly the '
                                 'PVLP151->GF edges of the pinned graph with the pinned body ids; every deletion row '
                                 'records weight_sha256 = deleted_weight_sha256 and zero_edges = the 8 indices; every '
                                 'other row records base_weight_sha256 and no zero_edges; parent_weight_unchanged true '
                                 'in every row; graph.npz unchanged.',
            'G5_shams': '(a) intact == frozen A2 counts (manifest pinned, every file hash verified); (b) clamp == '
                        'accepted clamp counts (manifest pinned, every file hash verified); (c) seed-0 empty-deletion '
                        'sham == intact, byte for byte.',
            'G6_complete_and_bound': 'exactly the 99 expected rows; provenance pins on every row (contract, graph, '
                                     'v3, cpu, engine, code = the frozen candidate given externally, this prereg); a '
                                     'pathA_shard.py exit record with exit_code 0 for every shard; full shard logs.',
            'on_failure': 'any of G1-G4, G5c or G6 fails: INVALID, no interpretation. G5a or G5b failure: disclosed '
                          'first; then only same-code contemporaneous arms are compared.',
        },
        'report': {
            'kind': 'DESCRIPTIVE ONLY: no threshold, no verdict, no pathway fraction.',
            'per_row': 'absolute Hz of the six readouts for every seed and arm, plus the no-input baseline.',
            'paired': 'per seed and mean+-SD: deletion - intact, clamp - intact, deletion - clamp, for GF and every '
                      'other readout; seeds lower/higher/equal; percent only with a nonzero denominator.',
            'all_neuron_controls': 'per arm pair and seed: neurons whose count differs and summed absolute '
                                   'difference, overall and outside the stimulated set, PVLP151 and GF.',
            'interpretation_limit': 'Arms differ by network-wide consequences in a nonlinear recurrent model. Neither a '
                                    'similar nor a different GF change under deletion and clamp shows that indirect '
                                    'routes are necessary or unnecessary, or what share any route carries. No '
                                    'biological mediation is inferred.',
            'analysis': 'scripts/pathA_gf_edge_analyse.py (external pins; exit 2 if INVALID)',
        },
        'run_procedure': {
            'launcher': 'one scripts/pathA_shard.py per shard: --exit-file <shard>.exit.json --log <shard>.log -- '
                        'python scripts/pathA_run.py --contract ... --contract-sha256 <contract> --cf-silence-prereg '
                        '<this file> --cf-silence-prereg-sha256 <this sha> --out <shard> --only <ids>',
            'shards': 'one per driven base condition (LC15, LC4 + LC4_0, LPLC2); OMP/BLAS/NUMBA = 1, '
                      'CUDA_VISIBLE_DEVICES empty, taskset one core each',
            'evidence': 'new evidence directory; originals and accepted run read-only',
        },
        'compute': {
            'device': 'CPU, v3 numba reference kernel, 1 thread per process; no GPU',
            'rows': '99 rows: 72 driven (3 conditions x 3 arms x 8 seeds) + 3 seed-0 shams + 24 no-input rows (about 0 s)',
            'per_run_wall_s': 'about 27-31 s (accepted run on this host, 3 shards) plus about 1 s per row for the '
                              'weight copy and hashes',
            'estimate_core_hours': 0.65,
            'estimate_wall': 'about 14 min in 3 shards',
            'budget_wall_hours': 1.0,
            'stop_rule': 'stop at the budget or any integrity failure; keep partial data; never extrapolate. Fresh '
                         'resource check first; at most 3 single-thread shards; protected services untouched.',
        },
    }
    out = Q / 'pvlp151_gf_edge_prereg.json'
    out.write_text(json.dumps(p, indent=1) + '\n')
    print(sha(out))


if __name__ == '__main__':
    main()
