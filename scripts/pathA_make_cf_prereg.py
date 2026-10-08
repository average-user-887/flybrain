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
RATES = [25, 50, 100, 150, 200]
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
            'PSI': cells('^PSI$', 'peripherally synapsing interneuron (GF->PSI 16, PSI->DLMn 449 contacts)')}
    by_id = {c['id']: c for c in contract['conditions']}
    base_ids = [f'A2_LC4_{r}' for r in RATES] + [f'A2_LPLC2_{r}' for r in RATES] + \
               ['A2_IRR_LC15_100', 'A2_IRR_LC15_200']
    conds = []
    for b in base_ids:
        assert b in by_id and not by_id[b].get('silence') and not by_id[b].get('counterfactual')
        conds.append(dict(id=f'{b}_CF_silence_PVLP151', base_condition=b, add_silence=['PVLP151'],
                          counterfactual='silence_PVLP151', seeds=contract['protocol']['seeds']))
    for b in ['A2_LC4_200', 'A2_LPLC2_200', 'A2_IRR_LC15_200']:
        conds.append(dict(id=f'{b}_REPRO', base_condition=b, add_silence=[], counterfactual=None, seeds=[0]))

    prereg = {
        'schema': 'neurofly.pathA.cf_prereg/1',
        'title': 'A2 PVLP151 existing-wiring counterfactual: does the LC15 control drive onto GF depend on LC15->PVLP151->GF?',
        'label': LABEL,
        'frozen_before_any_counterfactual_run': True,
        'status': 'PREPARED, NOT RUN. Decision rule proposed by Claude; Root (Astra) accepts or replaces it before '
                  'the run, and any change gives a new prereg sha256.',
        'frozen_contract_sha256': csha,
        'graph_npz_sha256': contract['data']['graph_npz_sha256'],
        'neurons_feather_sha256': contract['data']['neurons_feather_sha256'],
        'not_a_fix': 'The original A2 FAIL (contract cd69cb6d, results of code 6bc971f) is unchanged by any outcome. '
                     'Nothing here re-grades A2 or tunes the model.',
        'owner_rules': contract['owner_rules'] + ['no edge added: no gap junction, no invented connection'],
        'method': {
            'silencing': 'the frozen contract method, unchanged: constant silence_drive '
                         f"{contract['protocol']['silence_drive']} on the PVLP151 cells for the whole run "
                         '(membrane held at the E_inh floor, no spikes). v3 transmits only spikes, so the clamp '
                         'is equivalent to zeroing exactly the outgoing existing edges of the 4 PVLP151 cells '
                         '(1752 edges); their incoming edges and every other edge, weight and the graph file '
                         'are untouched (tests/test_pathA_cf_silence.py).',
            'runner': 'scripts/pathA_run.py with the frozen contract plus --cf-silence-prereg/--cf-silence-prereg-sha256; '
                      'without the flag the runner path is unchanged.',
            'inputs': 'identical to A2: same activated sets, rates, seeds and Poisson RNG '
                      '(default_rng([seed, round(10*rate), n_activated])), so each CF row is paired with the '
                      'frozen intact row of the same base condition and seed.',
            'intact_reference': 'the frozen A2 rows (336-run evidence, code 6bc971f, engine 473946a0...803e); '
                                'PVLP151/DLMn/PSI intact rates come from its stored per-run all-neuron counts '
                                '(sha-of-shas of the A2 count files 4bb7ffe4cecfb91e7cc116db77851cddbc0f6d753481ef3a1c115c1c78f79943).',
        },
        'sets': sets,
        'readouts': ['GF', 'TTMn', 'PVLP151', 'DLMn', 'PSI'],
        'conditions': conds,
        'validity_gates': {
            'V1_clamp_holds': 'PVLP151 spike count is 0 in every CF row.',
            'V2_reproduction': 'each _REPRO row (intact, new code) has all-neuron spike counts bit-identical to the '
                               'frozen A2 row of the same condition and seed 0; otherwise the code path changed and '
                               'nothing is concluded.',
            'V3_complete': 'all 96 CF rows present, each bound to contract cd69cb6d, graph 4b2f87cc, dynamics v3, '
                           'backend cpu, the frozen A2 engine sha256 and this prereg sha256.',
            'on_failure': 'INVALID: report the failing gate, no verdict.',
        },
        'predictions_before_run': [
            'Q: does the LC15 irrelevant-cell control drive onto GF depend on the LC15->PVLP151->GF route?',
            'Primary expectation (Claude): SUPPORTED. LC15 has no direct GF contact and PVLP151 is its named '
            'anatomical route to GF; the counter-hypothesis is that other polysynaptic LC15->...->GF routes carry '
            'most of the drive, which would give PARTIAL or REJECTED.',
            'Secondary expectations (reported only): LC4->GF changes little (PVLP151 is ~2 Hz under LC4 intact); '
            'LPLC2->GF falls (PVLP151 fires ~192 Hz under LPLC2 intact); TTMn, DLMn and PSI changes are reported '
            'without a directional prediction.',
        ],
        'decision_rule': {
            'metric': 'M_r = seed-mean GF rate under A2_IRR_LC15_r_CF_silence_PVLP151 / seed-mean GF rate under the '
                      'frozen A2_IRR_LC15_r, r in {100, 200}; paired per-seed sign of (CF - intact).',
            'SUPPORTED': 'M_100 <= 0.5 and M_200 <= 0.5, and GF is lower under the clamp in 8/8 seeds at both rates: '
                         'the PVLP151 route carries the majority of the LC15 drive onto GF.',
            'REJECTED': 'M_100 >= 0.8 and M_200 >= 0.8: the route carries at most a fifth of it; LC15 drives GF '
                        'mainly through other existing routes.',
            'PARTIAL': 'neither SUPPORTED nor REJECTED, and GF is lower under the clamp in 8/8 seeds at both rates.',
            'INCONCLUSIVE': 'anything else.',
            'threshold_basis': '0.5 = majority of the drive; 0.8 = at most a fifth. Chosen as generic fractions, not '
                               'from any counterfactual output (none exists). The intact A2 values were already known '
                               'when this was written; the earlier proposal of "<= 24 Hz" was withdrawn as post hoc and '
                               'is not used.',
            'reported_not_graded': [
                'control ratio under the clamp GF(LC15_r CF) / GF(LC4_r CF) and TTMn likewise, against the frozen '
                'contract limit 0.1 (reused, not new); labelled counterfactual, never an A2 re-grade',
                'paired CF/intact ratios for GF, TTMn, PVLP151, DLMn, PSI under every condition, mean and SD over seeds',
            ],
            'analysis': 'scripts/pathA_cf_analyse.py, committed with this prereg.',
        },
        'compute': {
            'device': 'CPU, numba v3 reference kernel, 1 thread per process; no GPU',
            'runs': '96 CF (12 conditions x 8 seeds) + 3 reproduction rows = 99 runs of 1000 ms',
            'per_run_wall_s': 'about 35-40 s (frozen A2 LC conditions: 22.6-40.6 s mean per condition, 4 processes '
                              'sharing a CPU)',
            'estimate_core_hours': 1.1,
            'estimate_wall': 'about 66 min in 1 process, about 17 min in 4 shards',
            'budget_wall_hours': 2.0,
            'stop_rule': 'stop when the budget is exhausted; report completed conditions only, never extrapolate. '
                         'Start only when cores are free and Root approves; not to be run alongside the S3 runs.',
        },
    }
    Path(args.out).write_text(json.dumps(prereg, indent=1) + '\n')
    print(hashlib.sha256(Path(args.out).read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
