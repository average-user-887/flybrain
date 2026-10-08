#!/usr/bin/env python3
"""Path A: write the PVLP151 existing-wiring counterfactual preregistration (no run).

LABELLED OFFLINE COUNTERFACTUAL, not NeuroFly wiring and not a fix: the frozen A2
conditions are repeated with the 4 PVLP151 cells clamped by the contract's own
silencing method (constant ``silence_drive``; no weight, edge or graph change).  In
the v3 spiking engine a clamped cell never spikes, so exactly its outgoing existing
edges never transmit.  No edge is added.  The original A2 FAIL is unchanged.

Written once and committed before any counterfactual run; the runner refuses a
prereg whose sha256 differs from the value given with ``--cf-silence-prereg-sha256``.
Nothing here reads simulation output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'qualification/pathA/contract.json'
OUT = ROOT / 'qualification/pathA/pvlp151_cf_prereg.json'
BASE_IDS = ['A2_IRR_LC15_200', 'A2_LC4_200', 'A2_LPLC2_200', 'A2_LC4_0']
LABEL = 'LABELLED OFFLINE COUNTERFACTUAL: PVLP151 clamped (existing wiring only, no edge added, not a fix)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    contract = json.loads(CONTRACT.read_text())
    csha = hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    import pyarrow.feather as feather
    cdir = Path(os.environ['NEUROFLY_CONNECTOME_DIR'])
    n = feather.read_table(cdir / 'normalized/neurons.feather',
                           columns=['node_index', 'source_id', 'cell_type']).to_pandas()
    ct = n.cell_type.fillna('')

    def cells(rule, note):
        m = ct.str.match(rule).to_numpy()
        return dict(node_index=[int(i) for i in n.node_index[m]], body_ids=[int(b) for b in n.source_id[m]],
                    rule=dict(type=rule), note=note, count=int(m.sum()))

    sets = {'PVLP151': cells('^PVLP151$', 'shared LC hub; LC15->PVLP151 1689 and PVLP151->GF 603 contacts (A2 trace)'),
            'DLMn': cells('^DLMn', 'dorsal longitudinal flight-muscle motor neurons (types "DLMn a, b", "DLMn c-f")'),
            'PSI': cells('^PSI$', 'peripherally synapsing interneuron (GF->PSI 16, PSI->DLMn 449 contacts)'),
            'GFC2': cells('^GFC2$', 'giant-fibre-coupled interneuron 2 (GF->GFC2 144, GFC2->TTMn 481 contacts)')}
    by_id = {c['id']: c for c in contract['conditions']}
    seeds = contract['protocol']['seeds']
    conds = []
    for b in BASE_IDS:
        assert b in by_id and not by_id[b].get('silence') and not by_id[b].get('counterfactual')
        conds.append(dict(id=f'{b}_REPLAY', base_condition=b, add_silence=[], counterfactual=None, seeds=seeds,
                          role='sham intact replay; must reproduce the frozen stored counts bit for bit'))
        conds.append(dict(id=f'{b}_CF_silence_PVLP151', base_condition=b, add_silence=['PVLP151'],
                          counterfactual='silence_PVLP151', seeds=seeds,
                          role='PVLP151 output silenced by the frozen clamp'))
        if b != 'A2_LC4_0':
            conds.append(dict(id=f'{b}_CF_zero_out_PVLP151', base_condition=b, add_silence=[],
                              zero_outgoing=['PVLP151'], counterfactual='zero_outgoing_PVLP151', seeds=[0],
                              role='in-run audit: outgoing PVLP151 weights 0 in an in-memory copy; non-PVLP151 '
                                   'counts must equal the clamp row of the same seed'))

    prereg = {
        'schema': 'neurofly.pathA.cf_prereg/2',
        'title': 'A2 PVLP151 existing-wiring counterfactual: does the observed LC15 response depend on the existing '
                 'PVLP151 outputs?',
        'label': LABEL,
        'scope': 'ASTRA_NEXT_GF_TTMN_CARD_DRAFT_20261008 and ASTRA_PVLP151_PREPARATION_REVIEW_20261008',
        'frozen_before_any_counterfactual_run': True,
        'status': 'PREPARED, NOT RUN. No simulation is authorised by this file; it runs only after the current '
                  'S3/replay assessment, an independent source/ID/intervention audit and Root GO.',
        'frozen_contract_sha256': csha,
        'graph_npz_sha256': contract['data']['graph_npz_sha256'],
        'neurons_feather_sha256': contract['data']['neurons_feather_sha256'],
        'not_a_fix': 'A model counterfactual, not proof of biological mediation and not a repair of A2. The original '
                     'A2 FAIL (contract cd69cb6d) and the unresolved electrical GF-TTMn wiring are unchanged by any '
                     'outcome. Nothing here re-grades A2, fits physiology or selects stimuli from outcomes.',
        'owner_rules': contract['owner_rules'] + ['no edge added: no gap junction, no invented connection',
                                                  'the graph file and stored A2 results are never altered'],
        'method': {
            'silencing': 'the frozen contract method, unchanged: constant silence_drive '
                         f"{contract['protocol']['silence_drive']} on the 4 PVLP151 cells for the whole run "
                         '(membrane held at the E_inh floor, no spikes). v3 transmits only spikes, so this removes '
                         'exactly the transmission of their 1752 outgoing existing edges; incoming edges, every '
                         'other edge, all weights and the graph file are untouched.',
            'evidence_for_the_clamp': 'PROOF comes from the real run (gates G1, G4). Supporting only: '
                                      'tests/test_pathA_cf_silence.py (toy-graph clamp == zero-outgoing-edge '
                                      'equivalence; on the pinned graph only PVLP151 out-edges are affected and '
                                      'the sha256 is unchanged).',
            'runner': 'scripts/pathA_run.py with the frozen contract plus --cf-silence-prereg and '
                      '--cf-silence-prereg-sha256; without the flag the runner path is unchanged except for a final '
                      're-hash of graph.npz.',
            'duration_dt': 'existing: 1000 ms, dt 0.1 ms; reset to rest before every run.',
            'inputs': 'identical to A2: same activated sets, rate 200 Hz, seeds 0-7 and Poisson RNG '
                      '(default_rng([seed, round(10*rate), n_activated])). Each row records the sha256 of its exact '
                      'input train (activated indices + event matrix); intact replay, clamp and zero-out rows of '
                      'the same base condition and seed must carry the same value.',
            'pairing': 'clamp row vs intact replay row of the same base condition and seed (same code, same prereg); '
                       'the frozen stored A2 rows are used as controls only after the replay has reproduced them.',
            'provenance': 'every row carries contract, graph, dynamics, backend, engine and code sha plus this '
                          'prereg sha256; the code sha is the commit the run is started from and must be recorded '
                          'in the result.',
            'frozen_reference': 'stored A2 counts of code 6bc971f, engine 473946a0...803e; sha-of-shas of the A2 '
                                'count files 4bb7ffe4cecfb91e7cc116db77851cddbc0f6d753481ef3a1c115c1c78f79943.',
        },
        'sets': sets,
        'readouts': ['GF', 'TTMn', 'PVLP151', 'PSI', 'DLMn', 'GFC2'],
        'conditions': conds,
        'gates': {
            'G1_clamp_zero': 'PVLP151 spike count is 0 in every CF_silence row (from that row\'s own counts).',
            'G2_clean_initial_state': 'every row records initial_state_clean = true: after reset, membrane at rest, '
                                      'conductances, refractory, delay queue, active set and clocks all zero.',
            'G3_identical_inputs': 'replay, clamp and zero-out rows of one base condition and seed have the same '
                                   'input_sha256.',
            'G4_zero_outgoing_semantics': 'at seed 0, each zero-out row has all non-PVLP151 counts identical to the '
                                          'clamp row of the same condition.',
            'G5_replay_reproduces_frozen': 'every REPLAY row has all-neuron counts bit-identical to the stored '
                                           'frozen A2 row of the same condition and seed.',
            'G6_complete_and_bound': 'all rows present; provenance matches the contract, graph, frozen engine and '
                                     'this prereg; graph.npz unchanged after the run.',
            'on_failure': 'any of G1-G4 or G6 fails: INVALID, report the gate, no interpretation. If only G5 fails, '
                          'the intact replay (same code) is still the paired control, the frozen rows are NOT used, '
                          'and the mismatch is reported first.',
        },
        'report': {
            'kind': 'DESCRIPTIVE ONLY. No pass threshold and no SUPPORTED/REJECTED verdict; zero or contrary '
                    'effects are valid outcomes.',
            'per_row': 'absolute Hz for GF, TTMn, PVLP151, PSI, DLMn, GFC2 (mean over all cells of the set, silent '
                       'cells included) for every seed, intact replay and clamp.',
            'paired': 'paired difference (clamp - intact) per seed, its mean and SD over seeds, the number of seeds '
                      'with a decrease/increase/no change; percent change only where the intact denominator is '
                      'nonzero, stated explicitly.',
            'baseline': 'no-input condition A2_LC4_0, intact and clamped, reported alongside.',
            'interpretation_limit': 'Differences are TOTAL NETWORK CONSEQUENCES of removing PVLP151 output in a '
                                    'nonlinear recurrent network. They are not fractions of the response assigned '
                                    'to the LC15->PVLP151->GF pathway, and no pathway share is inferred.',
            'analysis': 'scripts/pathA_cf_analyse.py, committed with this prereg.',
            'deliverable': 'one concise report: pinned artifacts, gates, paired results, limitations, next '
                           'falsifiable question.',
        },
        'prior_expectation_not_a_criterion': 'Claude expects GF under LC15 to fall when PVLP151 is silenced (no '
                                             'direct LC15-GF contact); LC4->GF little change; LPLC2->GF lower. '
                                             'Recorded only so that it cannot be revised after the run.',
        'compute': {
            'device': 'CPU, numba v3 reference kernel, 1 thread per process; no GPU',
            'runs': '48 driven runs (3 conditions x 8 seeds x intact replay + clamp) + 3 zero-out audit runs at '
                    'seed 0 + 16 no-input baseline runs (about 0 s each) = 67 rows of 1000 ms',
            'per_run_wall_s': 'about 36-41 s for the driven @200 conditions (frozen A2: LC4 37.0, LPLC2 40.6, '
                              'LC15 35.9 s mean per run, 4 processes sharing the CPU)',
            'estimate_core_hours': 0.55,
            'estimate_wall': 'about 33 min in 1 process, about 11 min in 3 shards (one per base condition)',
            'budget_wall_hours': 1.0,
            'stop_rule': 'stop when the budget is exhausted; report completed rows only, never extrapolate. '
                         'Schedule only when it does not compete with S3/replay or protected services.',
        },
    }
    Path(args.out).write_text(json.dumps(prereg, indent=1) + '\n')
    print(hashlib.sha256(Path(args.out).read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
