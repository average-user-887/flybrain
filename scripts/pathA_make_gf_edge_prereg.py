#!/usr/bin/env python3
"""Path A: write the PROPOSED preregistration for the PVLP151->GF direct-edge-deletion
counterfactual (no run; root review first).  Reads only the frozen contract, the accepted
PVLP151 prereg and the read-only inventory (pvlp151_gf_inventory.json)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / 'qualification/pathA'
BASES = ['A2_IRR_LC15_200', 'A2_LC4_200', 'A2_LPLC2_200', 'A2_LC4_0']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    contract, cf = json.loads((Q / 'contract.json').read_text()), json.loads((Q / 'pvlp151_cf_prereg.json').read_text())
    inv = json.loads((Q / 'pvlp151_gf_inventory.json').read_text())
    edges = inv['direct_pvlp151_to_gf']['rows']
    seeds = contract['protocol']['seeds']
    conds = []
    for b in BASES:
        conds += [dict(id=f'{b}_REPLAY', base_condition=b, arm='intact', add_silence=[], zero_edges=[], seeds=seeds),
                  dict(id=f'{b}_DEL_PVLP151_GF', base_condition=b, arm='direct_edge_deletion', add_silence=[],
                       zero_edges=[e['edge_index'] for e in edges], seeds=seeds),
                  dict(id=f'{b}_CF_silence_PVLP151', base_condition=b, arm='all_output_clamp', add_silence=['PVLP151'],
                       zero_edges=[], seeds=seeds)]
        if b != 'A2_LC4_0':
            conds.append(dict(id=f'{b}_SHAM_DEL_EMPTY', base_condition=b, arm='sham_empty_deletion', add_silence=[],
                              zero_edges=[], weight_copy=True, seeds=[0]))
    p = {
        'schema': 'neurofly.pathA.cf_prereg/3',
        'title': 'PVLP151->GF direct-edge deletion vs all-output clamp: does removing only the existing PVLP151->GF '
                 'edges reproduce the GF change seen under the all-output clamp?',
        'label': 'LABELLED OFFLINE COUNTERFACTUAL: existing wiring only, no edge added, not a fix, not an A2 re-grade',
        'status': 'PROPOSAL FOR ROOT REVIEW. Not frozen, not runnable: the runner support listed under '
                  'required_runner_changes is NOT implemented yet. No simulation until Root reviews and GOes; any '
                  'amendment gives a new sha256.',
        'origin': 'ASTRA_PVLP151_FINAL_DECISION_20261008 items 8-9 (post-hoc question raised after the accepted '
                  'all-output clamp result; stated as such).',
        'frozen_contract_sha256': sha(Q / 'contract.json'),
        'accepted_clamp_prereg_sha256': sha(Q / 'pvlp151_cf_prereg.json'),
        'accepted_clamp_code_sha': '33110a4f3292ab3ce8a5c64ee8a17b2626e1cec1',
        'inventory_sha256': sha(Q / 'pvlp151_gf_inventory.json'),
        'graph_npz_sha256': contract['data']['graph_npz_sha256'],
        'neurons_feather_sha256': contract['data']['neurons_feather_sha256'],
        'not_a_fix': cf['not_a_fix'],
        'owner_rules': cf['owner_rules'],
        'arms': {
            'intact': 'frozen A2 condition, unchanged (sham replay)',
            'direct_edge_deletion': f"the {len(edges)} existing PVLP151->GF edges (indices in zero_edges, "
                                    f"{inv['direct_pvlp151_to_gf']['contacts']} contacts, all v3 '+') get weight 0 in an "
                                    'in-memory copy of the v3 weight array; every other edge, PVLP151 itself (not '
                                    'clamped) and the graph file are untouched',
            'all_output_clamp': 'the accepted intervention, unchanged: PVLP151 held by the frozen silence_drive',
            'sham_empty_deletion': 'seed 0 only: the deletion code path with an empty edge list on a weight copy; '
                                   'must equal the intact row bit for bit',
        },
        'direct_edges': edges,
        'sets': cf['sets'],
        'readouts': cf['readouts'],
        'conditions': conds,
        'gates': {
            'G1_clamp_zero': 'PVLP151 spike count 0 in every clamp row.',
            'G2_clean_state': 'every row records initial_state_clean = true AND a sha256 of the full post-reset '
                              'state (all state arrays + clocks), identical across all rows.',
            'G3_identical_inputs': 'all arms of one base condition and seed carry the same 64-hex input_sha256.',
            'G4_deletion_exact': 'per deletion row, the recorded sha256 of the weight array used equals the pinned v3 '
                                 'array with exactly the listed 8 indices set to 0 (recomputed offline); the listed '
                                 'indices map to (pre, post) in PVLP151 x GF in the pinned graph; graph.npz unchanged.',
            'G5_shams': '(a) intact rows reproduce the frozen A2 counts (32-file manifest 79b4fa30) bit for bit; '
                        '(b) clamp rows reproduce the accepted 33110a4 clamp counts bit for bit; (c) each '
                        'sham_empty_deletion row equals its intact row bit for bit.',
            'G6_complete_and_bound': 'all rows present; provenance pins (contract, graph, neurons, engine 473946a0, '
                                     'code, this prereg) on every row; captured exit code 0 for every shard process.',
            'on_failure': 'any of G1-G4, G5c or G6 fails: INVALID, no interpretation. G5a/G5b failure: disclosed '
                          'first; only same-code contemporaneous arms are compared.',
        },
        'report': {
            'kind': 'DESCRIPTIVE ONLY: no threshold, no "reproduces" verdict, no pathway fraction.',
            'per_row': 'absolute Hz for the six readouts for every seed and arm, plus the no-input baseline.',
            'paired': 'per seed and as mean+-SD: deletion - intact, clamp - intact, deletion - clamp; seeds '
                      'lower/higher/equal; percent only with a nonzero denominator.',
            'all_neuron_controls': 'for each arm pair and seed: number of neurons whose spike count differs, summed '
                                   'absolute count difference, and the same restricted to neurons outside the '
                                   'stimulated set, PVLP151 and GF.',
            'interpretation_limit': 'Arms differ by network-wide consequences in a nonlinear recurrent model. A '
                                    'similar GF change under deletion and clamp would not show that the direct edges '
                                    'carry a fixed share, nor that indirect routes are absent; a difference would not '
                                    'identify which indirect route matters. No biological mediation is inferred.',
            'context_not_criterion': 'Inventory: direct PVLP151->GF = 603 contacts, 3.4 % of PVLP151 output '
                                     'contacts; 382 two-hop intermediates (135 types) with signs of both kinds; GF->PVLP151 '
                                     'feedback 18 contacts.',
        },
        'required_runner_changes': [
            'prereg condition field zero_edges: edge indices whose v3 weight is set to 0 in an in-memory copy; the '
            'runner verifies each index lies in PVLP151 x GF and refuses otherwise (weight_copy=true with an empty '
            'list exercises the same path for the sham).',
            'per-row record of sha256(weight array used) and sha256(full post-reset state) (Astra limit: reset '
            'evidenced only by assertions).',
            'launcher wrapper that captures each shard exit code and end time to <shard>.exit (Astra limit: exit '
            'codes not captured).',
            'analyser extension for the three arms, the shams and the all-neuron controls; unit tests incl. negative '
            'fixtures. No engine file changes.',
        ],
        'compute': {
            'device': 'CPU, v3 numba reference kernel, 1 thread per process; no GPU',
            'rows': f'{sum(len(c["seeds"]) for c in conds)} rows: 72 driven (3 conditions x 3 arms x 8 seeds) + '
                    '3 seed-0 shams + 24 no-input rows (about 0 s)',
            'per_run_wall_s': 'about 27-31 s (accepted run on this host, 3 shards)',
            'estimate_core_hours': 0.62,
            'estimate_wall': 'about 13 min in 3 shards (one per driven condition)',
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
