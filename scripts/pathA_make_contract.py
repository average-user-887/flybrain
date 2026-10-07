#!/usr/bin/env python3
"""Path A, phase 2: write the frozen experiment contract from the phase-1 mapping.

The contract is written ONCE and committed before any experimental run.  The
runner (scripts/pathA_run.py) refuses to run unless the contract's sha256
matches the frozen value; the analysis (scripts/pathA_analyse.py) reads its
criteria from the same file.  Nothing here looks at simulation output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEVELS = [0, 25, 50, 100, 150, 200]
CAL_LEVELS = [0, 10, 25, 50, 100, 150, 200]
SEEDS = list(range(8))
IRRELEVANT_SEED = 20261008


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget-hours', type=float, required=True)
    ap.add_argument('--out', default=str(ROOT / 'qualification/pathA/contract.json'))
    args = ap.parse_args()
    mapping = json.loads((ROOT / 'qualification/pathA/mapping.json').read_text())
    keep = ['JO-CE', 'JO-F', 'aBN1', 'aDN1', 'aDN2', 'ProLN-MN', 'LC4', 'LPLC2', 'GF', 'TTMn',
            'sugarGRN-cal', 'MN9']
    sets = {k: dict(node_index=mapping['sets'][k]['node_index'], body_ids=mapping['sets'][k]['body_ids'],
                    rule=mapping['sets'][k]['rule'], note=mapping['sets'][k]['note'],
                    count=mapping['sets'][k]['count']) for k in keep}

    # Irrelevant-cell control sets, chosen by declared anatomical rules BEFORE any run.
    import pyarrow.feather as feather
    cdir = Path(os.environ['NEUROFLY_CONNECTOME_DIR'])
    n = feather.read_table(cdir / 'normalized/neurons.feather',
                           columns=['node_index', 'source_id', 'cell_type']).to_pandas()
    a = feather.read_table(cdir / 'annotations.feather', columns=['bodyId', 'class']).to_pandas() \
        .drop_duplicates('bodyId').set_index('bodyId')
    n = n.join(a, on='source_id')
    orn = np.flatnonzero((n['class'].fillna('') == 'olfactory').to_numpy())
    rng = np.random.default_rng(IRRELEVANT_SEED)
    pick = np.sort(rng.choice(orn, size=sets['JO-CE']['count'], replace=False))
    sets['IRR-ORN'] = dict(node_index=[int(i) for i in pick], body_ids=[int(b) for b in n.source_id.iloc[pick]],
                           rule=dict(annotation_class='olfactory', sample=f'uniform without replacement, '
                                     f'size = |JO-CE|, numpy default_rng({IRRELEVANT_SEED})'),
                           note='A1 irrelevant-cell control: olfactory receptor neurons (same antenna, other modality)',
                           count=int(len(pick)))
    lc15 = np.flatnonzero((n.cell_type.fillna('') == 'LC15').to_numpy())
    sets['IRR-LC15'] = dict(node_index=[int(i) for i in lc15], body_ids=[int(b) for b in n.source_id.iloc[lc15]],
                            rule=dict(type='^LC15$'),
                            note='A2 irrelevant-cell control: LC15 (126 cells = |LC4|; 0 direct synapses onto GF in the mapping)',
                            count=int(len(lc15)))
    joab = np.flatnonzero(n.cell_type.fillna('').str.match(r'^JO-(A|B)').to_numpy())
    sets['TIMING-JOAB'] = dict(node_index=[int(i) for i in joab], body_ids=[int(b) for b in n.source_id.iloc[joab]],
                               rule=dict(type='^JO-(A|B)'), note='harness timing probe only; never a readout or evidence',
                               count=int(len(joab)))

    conds = []

    def add(test, cid, activate, rate, **kw):
        conds.append(dict(id=cid, test=test, activate=activate, rate_hz=rate, **kw))

    for r in LEVELS:
        add('A1', f'A1_JOCE_{r}', ['JO-CE'] if r else [], r)
    for r in LEVELS[1:]:
        add('A1', f'A1_JOF_{r}', ['JO-F'], r)
    for r in (100, 200):
        add('A1', f'A1_JOCE_{r}_silence_aBN1', ['JO-CE'], r, silence=['aBN1'])
        add('A1', f'A1_IRR_ORN_{r}', ['IRR-ORN'], r)
    add('A1', 'A1_JOCE_200_SHUFFLED', ['JO-CE'], 200, counterfactual='shuffled_input')
    for r in LEVELS:
        add('A2', f'A2_LC4_{r}', ['LC4'] if r else [], r)
    for r in LEVELS[1:]:
        add('A2', f'A2_LPLC2_{r}', ['LPLC2'], r)
    for r in (100, 200):
        add('A2', f'A2_LC4_{r}_silence_GF', ['LC4'], r, silence=['GF'])
        add('A2', f'A2_LPLC2_{r}_silence_GF', ['LPLC2'], r, silence=['GF'])
        add('A2', f'A2_IRR_LC15_{r}', ['IRR-LC15'], r)
    add('A2', 'A2_LC4_200_SHUFFLED', ['LC4'], 200, counterfactual='shuffled_input')
    add('A2', 'A2_LPLC2_200_SHUFFLED', ['LPLC2'], 200, counterfactual='shuffled_input')
    for r in CAL_LEVELS:
        add('CAL', f'CAL_SUGAR_{r}', ['sugarGRN-cal'] if r else [], r)

    mono = dict(rule='seed-mean rate at each successive level >= previous level - max(0.5 Hz, 0.10 * previous)')
    contract = dict(
        schema='neurofly.pathA.contract/1',
        title='Path A known-circuit battery: JO -> antennal grooming; LC4/LPLC2 -> GF -> TTMn; sugar -> MN9 (calibration)',
        frozen_before_any_run=True,
        owner_rules=['no invented connections', 'no fitting to the behaviour under test',
                     'engine equations, weights and model unchanged', 'a FAIL is reported, not tuned'],
        data=dict(dataset='MaleCNS v1.0', graph_npz_sha256=mapping['graph_npz_sha256'],
                  neurons_feather_sha256=mapping['neurons_feather_sha256'],
                  annotations_feather_sha256=mapping['annotations_feather_sha256'],
                  electrical_synapses='absent from the data; not modelled'),
        protocol=dict(
            dynamics='v3', backend='cpu', transmitter_policy='v3-modulatory-only (unclear=excitatory), as the engine default',
            synaptic_scale_mV_per_contact=0.275,
            synaptic_scale_provenance='Shiu et al. 2024 single free parameter, chosen on sugar GRN -> MN9; hence CAL is calibration, not evidence',
            dt_ms=0.1, duration_ms=1000.0, initial_state='rest (-52 mV), all transients zero, reset before every run',
            activation='independent Bernoulli-per-0.1 ms-tick Poisson events per activated cell at rate_hz; each event = one tick of pulse_drive '
                       '(forces a spike unless refractory), equivalent in effect to Shiu 2024 Poisson synapse 250 x W_syn',
            pulse_drive=10000.0, silence_drive=-10000.0,
            silencing='constant silence_drive for the whole run: membrane held at the E_inh floor (-70 mV), no spikes; no weight changed',
            shuffled_input='LABELLED OFFLINE COUNTERFACTUAL ONLY, not NeuroFly wiring: targets of every out-edge of the activated cells '
                           'replaced by uniform random neuron indices (numpy default_rng(10000 + seed)); weights and all other edges unchanged',
            readout_window_ms=[0.0, 1000.0],
            metric='firing rate = spike count in the window / 1.0 s per cell, averaged over the cells of the readout set; '
                   'summarised as mean and SD over seeds',
            seeds=SEEDS, seed_scope='seeds set the Poisson event trains only; the network is deterministic',
            poisson_rng='numpy default_rng([seed, round(10*rate_hz), n_activated_cells]); silencing conditions therefore receive '
                        'exactly the same input trains as their unsilenced pair',
            timing_probe_set='TIMING-JOAB',
            compute=dict(device='CPU (numba v3 reference kernel); no GPU job', budget_wall_hours=args.budget_hours,
                         stop_rule='stop when the budget is exhausted; report completed conditions only, never extrapolate')),
        sets=sets,
        tests=dict(
            A1=dict(
                kind='BLIND (NeuroFly never fitted to it)',
                readouts=['aBN1', 'aDN1', 'aDN2', 'ProLN-MN', 'JO-CE', 'JO-F'],
                predictions=[
                    'P1: aBN1 (SAD093) firing rises monotonically with the JO-CE activation rate and is robust (>= 10 Hz) at 100 and 200 Hz.',
                    'P2: JO-F activation does NOT robustly drive aBN1 despite 131 direct excitatory contacts: aBN1 rate under JO-F <= 0.5 x '
                    'its rate under JO-CE at 100 and at 200 Hz (anatomical basis: GABAergic GNG301/GNG516 cells receive JO-F input and synapse onto aBN1).',
                    'S1: aDN1 (DNg62) and aDN2 (DNge078) rise monotonically with the JO-CE rate.',
                    'S2: silencing aBN1 reduces the aDN1 and the aDN2 response to JO-CE by >= 50 % at 100 and 200 Hz.',
                    'S3: the front-leg (ProLN) motor-neuron population rises monotonically with the JO-CE rate, and is > 0 at 200 Hz for JO-CE and for JO-F.',
                ],
                criteria=dict(
                    P1=dict(readout='aBN1', series='A1_JOCE_{r}', levels=LEVELS, monotone=mono, min_rate_hz=10.0, at=[100, 200]),
                    P2=dict(readout='aBN1', numerator='A1_JOF_{r}', denominator='A1_JOCE_{r}', at=[100, 200], max_ratio=0.5),
                    S1=dict(readouts=['aDN1', 'aDN2'], series='A1_JOCE_{r}', levels=LEVELS, monotone=mono),
                    S2=dict(readouts=['aDN1', 'aDN2'], silenced='A1_JOCE_{r}_silence_aBN1', intact='A1_JOCE_{r}', at=[100, 200], max_ratio=0.5),
                    S3=dict(readout='ProLN-MN', series='A1_JOCE_{r}', levels=LEVELS, monotone=mono, positive_at=['A1_JOCE_200', 'A1_JOF_200']),
                    control_irrelevant=dict(readouts=['aBN1', 'aDN1', 'aDN2'], control='A1_IRR_ORN_{r}', reference='A1_JOCE_{r}', at=[100, 200],
                                            max_ratio=0.1, note='control VALID if every ratio <= 0.1'),
                    counterfactual_shuffled=dict(readout='aBN1', control='A1_JOCE_200_SHUFFLED', reference='A1_JOCE_200', max_ratio=0.5,
                                                 note='reported only; labelled offline counterfactual'),
                ),
                verdict_rule='PASS iff P1 and P2 hold and the irrelevant-cell control is valid; FAIL otherwise. '
                             'S1-S3 and the counterfactual are reported separately and do not change the verdict.',
                physiology_reference=[
                    'Shiu et al. 2024, Nature 634:210 (doi 10.1038/s41586-024-07763-9), Fig. 5h: calcium imaging, "JO-CE activated aBN1 '
                    'robustly, but JO-F neurons did not", matching their model prediction.',
                    'Hampel et al. 2015, eLife 4:e08758: JO-CE -> aBN1 -> aDN antennal-grooming command circuit (aDN1/aDN2 named there; '
                    'MaleCNS synonyms DNg62 / DNge078).',
                    'Hampel et al. 2020, eLife 9:e59976: activation of each JO subpopulation (incl. JO-CE and JO-F) elicits antennal grooming '
                    '(behaviour only; reference for S3, not a firing-rate measurement).'],
                blindness_note='Blind for NeuroFly (never fitted to it). Not novel: the same prediction was published by Shiu 2024 with a '
                               'current-based LIF on FlyWire; this is a reproduction on a different animal (MaleCNS) with v3 conductance dynamics.'),
            A2=dict(
                kind='BLIND (NeuroFly never fitted to it)',
                readouts=['GF', 'TTMn', 'LC4', 'LPLC2'],
                predictions=[
                    'P1: GF (DNp01) firing rises monotonically with LC4 rate and with LPLC2 rate and is >= 5 Hz at 200 Hz for each.',
                    'P2: TTMn follows GF: at 200 Hz (LC4 and LPLC2 separately) TTMn mean rate >= 0.5 x GF mean rate (biology: GF -> TTMn ~1:1 relay).',
                    'S1: silencing GF reduces the TTMn response to LC4 and to LPLC2 by >= 90 % at 100 and 200 Hz.',
                ],
                electrical_synapse_risk='GF -> TTMn is largely ELECTRICAL in biology (shakB gap junctions; mixed with a cholinergic component). '
                                        'Gap junctions are not in MaleCNS, so only the 90 chemical GF -> TTMn contacts (2 pairs) carry the relay '
                                        'here. P2 may fail for this data reason; it is still reported as FAIL.',
                criteria=dict(
                    P1=dict(readout='GF', series=['A2_LC4_{r}', 'A2_LPLC2_{r}'], levels=LEVELS, monotone=mono, min_rate_hz=5.0, at=[200]),
                    P2=dict(numerator_readout='TTMn', denominator_readout='GF', conditions=['A2_LC4_200', 'A2_LPLC2_200'], min_ratio=0.5),
                    S1=dict(readout='TTMn', pairs=[['A2_LC4_{r}_silence_GF', 'A2_LC4_{r}'], ['A2_LPLC2_{r}_silence_GF', 'A2_LPLC2_{r}']],
                            at=[100, 200], max_ratio=0.1),
                    control_irrelevant=dict(readouts=['GF', 'TTMn'], control='A2_IRR_LC15_{r}', reference='A2_LC4_{r}', at=[100, 200],
                                            max_ratio=0.1, note='control VALID if every ratio <= 0.1 (reference with zero rate counts as ratio 0 only if control is also 0)'),
                    counterfactual_shuffled=dict(readout='GF', pairs=[['A2_LC4_200_SHUFFLED', 'A2_LC4_200'], ['A2_LPLC2_200_SHUFFLED', 'A2_LPLC2_200']],
                                                 max_ratio=0.5, note='reported only; labelled offline counterfactual'),
                ),
                verdict_rule='PASS iff P1 and P2 hold and the irrelevant-cell control is valid; FAIL otherwise. Sub-results reported separately.',
                physiology_reference=[
                    'Ache et al. 2019, Curr Biol 29:1073: LC4 and LPLC2 synapse directly onto GF; optogenetic LPLC2 activation produces a '
                    'large rapid GF depolarisation; LC4 carries the velocity and LPLC2 the size component of the GF looming response.',
                    'von Reyn et al. 2014, Nat Neurosci 17:962: a GF spike drives the TTM-mediated short-mode takeoff.',
                    'Allen et al. 2006 / Phelan et al. 1996: GF -> TTMn is a mixed electrical (shakB) + chemical synapse.'],
                caveat='Direct LC activation is a circuit probe, not retinal looming detection. GF depolarisation is the biological '
                       'readout; the LIF proxy exposes only spikes.'),
            CAL=dict(
                kind='CALIBRATION CHECK - NOT EVIDENCE (0.275 mV was chosen by Shiu 2024 on this response)',
                readouts=['MN9', 'sugarGRN-cal'],
                predictions=['C1: MN9 rises monotonically with the sugar-GRN rate; at 100 Hz MN9 is 60-100 % of its maximum over levels (Shiu: ~80 %).'],
                criteria=dict(C1=dict(readout='MN9', series='CAL_SUGAR_{r}', levels=CAL_LEVELS, monotone=mono, at=100,
                                      fraction_of_max=[0.6, 1.0])),
                verdict_rule='CONSISTENT iff C1 holds; otherwise INCONSISTENT. Never counted as evidence.',
                identity_caveat='MaleCNS LB3b/LB3c/LB3d/LB4b carry flywireType "LB3"/"LB4b"; their split into LB3a-d is a MaleCNS-side '
                                'name match to the FlyWire v783 types of 20/21 Shiu root IDs. Identity is a name match, UNCERTAIN; '
                                'one side (rootSide R) only, 36 cells vs Shiu 21.'),
        ),
        conditions=conds,
        analysis='scripts/pathA_analyse.py reads runs.jsonl and applies exactly the criteria above; it is committed with this contract.',
    )
    Path(args.out).write_text(json.dumps(contract, indent=1) + '\n')
    print(len(conds), 'conditions x', len(SEEDS), 'seeds =', len(conds) * len(SEEDS), 'runs')


if __name__ == '__main__':
    main()
