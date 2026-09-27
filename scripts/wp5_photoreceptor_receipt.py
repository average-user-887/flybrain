"""Assemble docs/receipts/photoreceptor_encoder.json from the probe outputs.

    PYTHONPATH=. .venv/bin/python scripts/wp5_photoreceptor_receipt.py \
        outputs/wp5/photoreceptor-20260927 outputs/wp5/photoreceptor-einh60-20260927 \
        docs/receipts/photoreceptor_encoder.json

The receipt records the locked declaration's hash, every pin, the gate evaluation
by the verbatim §6.7 criteria, the stage-by-stage diagnostic of
docs/PHOTORECEPTOR_ENCODER.md §4, the predictions-versus-outcome and falsifier
tables, and what was therefore NOT run.  Nothing is recomputed from scratch here;
the probe JSON files are the evidence.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KEY_POPULATIONS = ('R1-R6_L', 'R1-R6_R', 'L1_L', 'L1_R', 'L2_L', 'L2_R', 'L3_L', 'L3_R',
                   'L5_L', 'L5_R', 'Mi1_L', 'Mi1_R', 'Mi9_L', 'Mi9_R', 'Tm1_L', 'Tm3_L',
                   'Tm9_L', 'T4a_L', 'T4a_R', 'T4b_L', 'T4b_R', 'T5a_L', 'T5b_L',
                   'HS_L', 'HS_R', 'H2_L', 'H2_R', 'DNa02_L', 'DNa02_R')


def config_block(path: Path) -> dict:
    d = json.loads((path / 'photoreceptor_probe.json').read_text())
    arms = {}
    for arm, a in d['arms'].items():
        g = a['gate']
        arms[arm] = dict(
            gate={k: g[k] for k in ('Q1', 'R1', 'R2')},
            gate_detail=dict(dna02_LminusR_per_direction=g['dna02_LminusR_per_direction'],
                             membrane_min_mV=g['membrane_min_mV'], gray=g['gray'],
                             stimulus=g['stimulus']),
            first_silent_stage=a['diagnostic']['first_silent_stage'],
            attribution=a['diagnostic']['attribution'],
            dsi=a['diagnostic']['dsi'],
            stage_table=[r for r in a['diagnostic']['table']],
            mean_yaw_rad_s={dirn: row['stimulus']['mean_yaw_rad_s']
                            for dirn, row in a['directions'].items()},
            sim_s_per_wall_s={dirn: row['sim_s_per_wall_s'] for dirn, row in a['directions'].items()},
        )
    return dict(
        source=str(path.relative_to(ROOT)) if path.is_absolute() else str(path),
        started_at=d['started_at'], finished_at=d['finished_at'], host=d['host'],
        lif_dynamics_version=d['lif_dynamics_version'], lif_dynamics_pin=d['lif_dynamics_pin'],
        lif_e_inh_mV=d['lif_e_inh_mV'], lif_dynamics_variant=d['lif_dynamics_variant'],
        lif_dynamics_variant_pin=d['lif_dynamics_variant_pin'],
        graph_sha256=d['graph']['graph_sha256'], policy_graph_sha256=d['policy_graph_sha256'],
        transmitter_policy=d['transmitter_policy'], io_map=d['io_map'], arms=arms)


def main():
    primary_dir, arm60_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    primary = json.loads((primary_dir / 'photoreceptor_probe.json').read_text())
    lock = ROOT / 'docs/receipts/photoreceptor_encoder_declaration.locked.md'
    receipt = dict(
        schema='neurofly.photoreceptor-encoder.v1',
        title='Photoreceptor-level encoder: does the connectome compute direction selectivity?',
        declaration_lock=dict(
            document='docs/PHOTORECEPTOR_ENCODER.md',
            locked_copy='docs/receipts/photoreceptor_encoder_declaration.locked.md',
            sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
            locked_at='2026-09-27T10:50:49+02:00',
            scope='Sections 0-6 (engine capability audit, the encoder, the predictions and '
                  'falsifiers, the diagnostic, the reused gate and metric, the prohibitions) were '
                  'written and hashed BEFORE any neuron was driven by this encoder. Sections 7+ are '
                  'filled in afterwards and do not alter 0-6; the locked copy is the proof.',
            forbidden='no change to i_max, noise, wavelength, temporal frequency, interommatidial '
                      'angle, contrasts, schedule, seeds, decoder, Q1/R1/R2 thresholds, transmitter '
                      'policy, E_inh or the preregistered TI metric to obtain a better reading'),
        status='complete at probe level; the confirmatory 24-run set was NOT run because R1 failed, '
               'which is the binding budget rule of docs/LIF_DYNAMICS_SPEC.md 6.7',
        pins=dict(
            photoreceptor_io_map='0931ec2fdc9c812af0477e3bc3e6b2077b07eedf61db8edbf4afeab1da943222',
            existing_optomotor_io_map_UNCHANGED='228c1b69251e73a3ec34d551202b98f8b6afddf723a77c89e3203053d2366c7b',
            dynamics_v3=primary['lif_dynamics_pin'],
            prereg_sha256=primary['prereg_sha256'],
            graph_sha256=primary['graph']['graph_sha256'],
            policy_graph_sha256=primary['policy_graph_sha256']),
        encoder=dict(module='brainlab/io_map_photoreceptor.py',
                     layout_rule=primary['layout_rule'], delta_phi_deg=primary['delta_phi_deg'],
                     driven=primary['io_map']['driven'], unresolved=primary['io_map']['unresolved'],
                     layout=primary['io_map']['layout'],
                     arms=dict(P1='primary, luminance-faithful: gray = 0.5*i_max on every '
                                  'resolved photoreceptor',
                               P2='declared arm, contrast-only: gray = 0 drive, as 12/13')),
        engine_capability_audit=dict(
            graded_or_threshold_linear_transmission='NONE. brainlab/engine.py advance, advance_v2 '
                                                    'and advance_v3 all transmit only on crossing '
                                                    'V_THRESHOLD_MV = -45 mV; a subthreshold '
                                                    'excursion transmits nothing.',
            only_continuous_input='the external drive array (the encoder itself)',
            temporal_parameters=dict(tau_m_ms=20.0, tau_syn_ms=5.0, delay_ms=1.8,
                                     refractory_ms=2.2, dt_ms=0.1,
                                     per_cell_type_kinetics=False, calcium_state=False,
                                     dendritic_compartments=False),
            inherited_from_source='Shiu et al. 2024 (Nature 634:210-219) state that their model '
                                  '"does not account for gap junctions, non-spiking neurons, '
                                  'internal state or long-range neuropeptides" and treat "each '
                                  'neuron identically as a spiking neuron". This engine reproduces '
                                  'a declared limitation of its source; it did not drop a '
                                  'mechanism the source had.'),
        gate_source=dict(criteria='docs/LIF_DYNAMICS_SPEC.md 6.7, code copied verbatim from '
                                  'outputs/wp5/v3-verify-20260926/gate.py into '
                                  'scripts/wp5_photoreceptor_probe.py',
                         probe_schedule='500 ms gray + 1 s rotation + 500 ms gray, contrast 1.0, '
                                        'seed 0, one run per direction (probe C, unchanged)'),
        configurations={'A-primary E_inh=-70': config_block(primary_dir),
                        'A-primary E_inh=-60 (declared v3-einh-60)': config_block(arm60_dir)},
        structural_in_weight_budget=primary['structural_in_weight_budget'],
        predictions_vs_outcome=[
            dict(prediction='P1 the first synapse is inhibitory into a silent network, so the '
                            'signal dies at or just after it',
                 outcome='CONFIRMED. Photoreceptors fire at 38.6 (L) / 39.6 (R) Hz per cell with '
                         '99.6 % / 98.4 % of cells firing; L1/L2/L3 are hyperpolarised to -64.8 / '
                         '-65.0 / -60.5 mV and their maximum membrane potential over the whole '
                         'rotation window is exactly V_rest = -52.00 mV, i.e. they never '
                         'depolarise at all. Nothing beyond the lamina ever leaves -52.00 mV.'),
            dict(prediction='P2 graded cells must spike or say nothing',
                 outcome='CONFIRMED in the stronger form: the lamina does not even respond '
                         'subthreshold in the depolarising direction, because the only fast input '
                         'it receives from the driven stage is inhibitory.'),
            dict(prediction='P3 no cell-type-specific delay, so DSI of order 0.05-0.2 at best',
                 outcome='NOT TESTABLE on this evidence: T4/T5 never fired, so their DSI is '
                         'undefined. The prediction is neither confirmed nor refuted.'),
            dict(prediction='P4 no calcium stage',
                 outcome='UNTESTED for the same reason.'),
            dict(prediction='the retina itself carries the stimulus',
                 outcome='CONFIRMED and worth stating: the photoreceptor mean rate is the same in '
                         'both directions (DSI +0.0003 / -0.0000), as any rate measure of a '
                         'full-field grating must be. The direction information is in the relative '
                         'phase across columns, which the drive does contain and which the '
                         'downstream correlator never sees.')],
        falsifiers=[
            dict(id='G1', statement='the photoreceptors themselves do not fire',
                 fired=False, evidence='38.6-39.6 Hz per cell, 98-100 % of cells firing; the '
                                       'derived drive scaling is adequate'),
            dict(id='G2', statement='the photoreceptors fire but nothing downstream does',
                 fired=True, evidence='every population beyond R1-R6 has 0 spikes in both '
                                      'directions in both arms and at both E_inh values'),
            dict(id='G3', statement='lamina and medulla respond but T4/T5 DSI ~ 0',
                 fired=False, evidence='not reached: the lamina does not respond'),
            dict(id='G4', statement='T4/T5 selective with the wrong sign', fired=False,
                 evidence='not reached'),
            dict(id='G5', statement='T4/T5 selective but selectivity lost before DNa02',
                 fired=False, evidence='not reached'),
            dict(id='G6', statement='the network runs away (Q1 fails)',
                 fired='partly',
                 evidence='arm P1 FAILS Q1: the gray screen is a lit gray screen, so 3,335 '
                          'photoreceptors fire tonically and the gray network rate is 1.24e5 '
                          'spikes/s, above the 1.0e5 threshold. It is the photoreceptors '
                          'themselves, not a recurrent runaway: DNa02 is at 0 Hz and no other '
                          'population fires. Arm P2 PASSES Q1 (gray = 0 drive = 0 spikes/s). '
                          'The threshold was not relaxed.')],
        outcome_metric=dict(
            metric='preregistered TI of docs/wp5_optomotor_prereg.json, unchanged',
            confirmatory_set_run=False,
            reason='R1 failed in both arms and at both E_inh values; docs/LIF_DYNAMICS_SPEC.md 6.7 '
                   'binds: "If Q1 or R1 fails, the 24-run confirmatory set is not run. The failure '
                   'is the result."',
            probe_level_decoded_yaw='identically 0.000000 rad/s in every window of every run, '
                                    'because DNa02 never spikes; the implied TI is exactly 0',
            comparison=dict(wp5_12_v3_einh_minus70_TI='+0.0671 [+0.0524, +0.0829] (T4/T5 encoder)',
                            wp5_13_v3_einh_minus60_TI='+0.0452 [+0.0237, +0.0689] (T4/T5 encoder)',
                            photoreceptor_encoder_TI='0 (no spike reaches the decoder)',
                            reading='the +0.0671 and +0.0452 of 12 and 13 are produced by an '
                                    'encoder that injects the direction selectivity into T4/T5. '
                                    'With the same stimulus delivered to photoreceptors instead, '
                                    'this engine transmits nothing past the lamina, so those '
                                    'numbers measure the graph\'s routing of an already-'
                                    'directional signal and not a computation of direction '
                                    'selectivity by the connectome.')),
        what_would_be_needed=[
            'graded (or threshold-linear) transmission for the non-spiking pathway, as a NEW '
            'declared dynamics version with its own pre-registration and pin - not implemented '
            'here, deliberately (declaration 6)',
            'or a maintained baseline of activity that an inhibitory photoreceptor synapse can '
            'modulate downward, which contradicts the zero-basal-rate assumption inherited from '
            'Shiu et al. 2024',
            'and, only once the pathway conducts at all, a cell-type-specific temporal filter of '
            'order tens of ms, which this engine cannot express (one global tau_syn = 5 ms and a '
            '1.8 ms hop delay)'],
        unchanged=dict(
            io_map='brainlab/io_map.py and OPTOMOTOR_IO_PIN are byte-unchanged',
            prereg='docs/wp5_optomotor_prereg.json is byte-unchanged',
            dynamics='dynamics_pin("v3") is byte-unchanged; E_inh -60 is the already-declared '
                     'variant v3-einh-60 of docs/EINH_SENSITIVITY.md',
            wp5='docs/WP5_OPTOMOTOR.md 1-13 are unaltered; this program appends 14 only',
            engine='brainlab/engine.py is unaltered'),
        reproduce=[
            'NEUROFLY_GRAPH_DIR=<graph dir> PYTHONPATH=. .venv/bin/python '
            'scripts/wp5_photoreceptor_probe.py --out outputs/wp5/photoreceptor-20260927 '
            '--arm P1,P2 --dynamics v3',
            'NEUROFLY_GRAPH_DIR=<graph dir> PYTHONPATH=. .venv/bin/python '
            'scripts/wp5_photoreceptor_probe.py --out outputs/wp5/photoreceptor-einh60-20260927 '
            '--arm P1,P2 --dynamics v3 --e-inh -60',
            'PYTHONPATH=. .venv/bin/python scripts/wp5_photoreceptor_receipt.py '
            'outputs/wp5/photoreceptor-20260927 outputs/wp5/photoreceptor-einh60-20260927 '
            'docs/receipts/photoreceptor_encoder.json'])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + '\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
