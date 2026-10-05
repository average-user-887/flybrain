"""Assemble docs/receipts/lif_dynamics_v6a.json from the committed raw / derived files."""
import hashlib
import json

from brainlab import graded_release as gr
from brainlab.graph_identity import LIF_DYNAMICS_V6A, dynamics_pin

D = 'docs/receipts/v6a_raw'
lock = 'docs/receipts/graded_gain_v6a_declaration.locked.md'
unit = json.load(open(f'{D}/unit_level.json'))
ident = json.load(open(f'{D}/identity_real_graph.json'))
stage = {t: json.load(open(f'{D}/{t}.stage.json')) for t in ('v6a_primary', 'v6a_K0')}
ana = {t: json.load(open(f'{D}/{t}.analysis.json')) for t in ('v6a_primary', 'v6a_K0')}
meta = {t: json.load(open(f'{D}/{t}.meta.json')) for t in ('v6a_primary', 'v6a_K0')}
sub = json.load(open(f'{D}/posthoc_input_subset.json'))
v5a = {t: json.load(open(f'docs/receipts/v5_raw/{t}.analysis.json'))
       for t in ('v4_protocol9', 'v5_primary', 'v5_upper_S2')}

TRACE = ['R1-R6', 'L1', 'L2', 'L3', 'Mi1', 'Tm3', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm4', 'Tm9', 'T4', 'T5', 'HS']


def trace(eye):
    out = {}
    for k in TRACE:
        key = f'{k}_{eye}'
        row = {t: round(stage['v6a_primary']['reference'][t][key]['p90'], 4)
               for t in ('v4_protocol9', 'v5_primary', 'v5_upper_S2')}
        for t in ('v6a_primary', 'v6a_K0'):
            row[t] = round(stage[t]['stage'][key]['p90'], 4)
        out[k] = row
    return out


def timing(a, eye='R'):
    return {k.split('_')[0]: round(a['D4'][f'{k}'][ 'equivalent_delay_ms'], 1)
            for k in (f'{p}_{eye}' for p in ('L1', 'L2', 'L3', 'Mi1', 'Tm3', 'Mi4', 'Mi9', 'Tm1', 'Tm2',
                                             'Tm4', 'Tm9', 'T4a', 'T5a')) if k in a['D4']}


rec = dict(
    receipt='LIF dynamics v6a: calibrated graded-synapse transfer gain',
    spec='docs/LIF_DYNAMICS_SPEC.md#10',
    declaration_lock=lock,
    declaration_sha256=hashlib.sha256(open(lock, 'rb').read()).hexdigest(),
    declaration_sha256_cited_by_code=LIF_DYNAMICS_V6A['declaration_sha256'],
    dynamics_pin_v6a=dynamics_pin('v6a'),
    release_tables=gr.RELEASE,
    photoreceptor_gain_derivation=unit['derivation'],
    arms=dict(primary='v5-receptor-class kinetics + v6a-calibrated',
              K0='v4-equivalent kinetics + v6a-calibrated'),
    gpu={t: meta[t]['gpu'] for t in meta} | {'identity_real_graph': ident['gpu']},
    unit_level=dict(
        small_signal_gain=unit['a1'],
        dynamic_1p5Hz={k: v for k, v in unit['a3'].items()},
        default_class_small_signal_gain={k: {s: x['small_signal_gain'] for s, x in v.items()}
                                         for k, v in unit['a4'].items()},
        verdict='P1 held: -6.0 under v6a-calibrated (declared -6 +- 2%), -0.147 under v5-linear, both arms'),
    real_graph_identity=dict(passed=ident['passed'], spikes=ident['spikes'], tolerance_mV=ident['tolerance_mV'],
                             v5_vs_v6a_v5linear_max_dV=ident['v5_primary_vs_v6a_v5linear']['v']['max_abs_diff'],
                             v5_vs_v5_repeat_max_dV=ident['v5_primary_vs_v5_repeat']['v']['max_abs_diff']),
    stage_gate={t: dict(gate=stage[t]['gate'], health={k: v for k, v in stage[t]['health'].items()
                                                       if k != 'other_windows'},
                        health_other_windows=stage[t]['health']['other_windows'])
                for t in stage},
    per_stage_trace_right_eye_p90_mV=trace('R'),
    per_stage_trace_left_eye_p90_mV=trace('L'),
    posthoc_input_receiving_subset=dict(
        label=sub['label'], added=sub['added'], prompted_by=sub['prompted_by'], anatomy=sub['anatomy'],
        right_eye={k: {t: sub['runs'][t][f'{k}_R'] for t in sub['runs']}
                   for k in ('L1', 'L2', 'L3', 'Mi1', 'Tm3', 'Tm1', 'Tm2', 'Mi9', 'Mi4', 'Tm4', 'Tm9', 'T4', 'T5')}),
    flicker_delay_ms_right_eye={'v4_protocol9': timing(v5a['v4_protocol9']), 'v5_primary': timing(v5a['v5_primary']),
                                'v5_upper_S2': timing(v5a['v5_upper_S2']),
                                'v6a_primary': timing(ana['v6a_primary']), 'v6a_K0': timing(ana['v6a_K0'])},
    animal_timing_reference='Behnia et al. 2014 whole-cell: Mi1 peaks ~18 ms after Tm3; Tm1 ~13 ms after Tm2',
    dna02=dict({t: dict(gray=ana[t]['D6']['gray'], yaw_clause=ana[t]['D6']['yaw_clause'],
                        flicker=ana[t]['D6'].get('flicker')) for t in ana}),
    hs_sign_test={t: ana[t]['D5'].get('sign_test') for t in ana},
    motion_gate='NOT RUN: the declared stage gate failed in both arms (spec §10.8 stop rule)',
    verdict=dict(
        stage_gate_primary='FAIL (T4/T5 clause: right-eye p90 T4 0.455, T5 0.482 mV < 1 mV; health: DNa02_R silent '
                           'in gray, 0 Hz). Lamina clause passed (L1 9.71, L2 3.80 mV).',
        stage_gate_K0='FAIL (T4/T5 clause: T4 0.442, T5 0.504 mV). Lamina and health passed.',
        motion_gate='not run',
        answer='Restoring the one measured synaptic gain brings the lamina to a biological scale and raises '
               'T4/T5 about 5x, but T4/T5 stay sub-millivolt (0.45-0.50 mV p90, 0.27-0.29 mV median on the '
               'input-receiving subset): the signal still attenuates ~0.1-0.4x per graded hop downstream of '
               'the lamina, where no transfer has been measured and v6a keeps v4\'s line. Whether the '
               'connectome then computes direction was not tested, because the stage gate failed.'),
    falsifiers=dict(F1='not triggered (unit probe -6.0; v5-linear reproduces v5)',
                    F2='not triggered (lamina in band in-network)',
                    F3='TRIGGERED in the primary only: DNa02_R silent in gray (health clause); K0 healthy',
                    F4='not triggered (T4/T5 below band, as predicted by P3)',
                    F5='not reached', F6='not reached',
                    F7='not triggered (both arms fail the stage gate, on the same clause)'),
)
json.dump(rec, open('docs/receipts/lif_dynamics_v6a.json', 'w'), indent=1)
print(json.dumps(rec['per_stage_trace_right_eye_p90_mV'], indent=0))
print(json.dumps(rec['flicker_delay_ms_right_eye'], indent=0))
