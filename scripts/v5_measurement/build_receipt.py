"""Evaluate the declared predictions P1-P7 and falsifiers F1-F8 of
docs/LIF_DYNAMICS_SPEC.md §9.7/§9.8 and write docs/receipts/lif_dynamics_v5.json.
Pure post-processing of the committed raw and analysis files."""
import hashlib
import json
import os

import numpy as np

R = 'docs/receipts/v5_raw/'
LOCK = 'docs/receipts/receptor_kinetics_v5_declaration.locked.md'


def load(tag):
    p = R + tag + '.analysis.json'
    return json.load(open(p)) if os.path.exists(p) else None


v4, v5, s2 = load('v4_protocol9'), load('v5_primary'), load('v5_upper_S2')
unit = json.load(open(R + 'unit_level.json'))
s0 = json.load(open(R + 's0_real_graph.json'))
speed = json.load(open(R + 'speed_probe.json'))
meta = {t: json.load(open(R + t + '.meta.json')) for t in ('v4_protocol9', 'v5_primary', 'v5_upper_S2')
        if os.path.exists(R + t + '.meta.json')}
T45 = [f'{b}{s}_{e}' for b in ('T4', 'T5') for s in 'abcd' for e in 'LR']


def p3(ref, run, scale=1.0):
    """P3 bounds: D1 curve within 0.05 mV per direction, D2 median within 0.05, D4 phase within 5 deg."""
    out = dict(d1_max_abs_diff_mV={}, d2_median_diff={}, d4_phase_diff_deg={})
    for k, d in run['D1'].items():
        if d['unit'] != 'mV':
            continue
        out['d1_max_abs_diff_mV'][k] = max(abs(d['tuning'][t] - ref['D1'][k]['tuning'][t]) for t in d['tuning'])
    for k, d in run['D2'].items():
        out['d2_median_diff'][k] = d['median_cell_abs_dsi_f1'] - ref['D2'][k]['median_cell_abs_dsi_f1']
    for k, d in run['D4'].items():
        diff = (d['phase_lag_deg'] - ref['D4'][k]['phase_lag_deg'] + 180) % 360 - 180
        out['d4_phase_diff_deg'][k] = diff
    viol = dict(
        d1=sorted(k for k, v in out['d1_max_abs_diff_mV'].items() if v >= 0.05 * scale),
        d2=sorted(k for k, v in out['d2_median_diff'].items() if abs(v) >= 0.05 * scale),
        d4=sorted(k for k, v in out['d4_phase_diff_deg'].items() if abs(v) >= 5.0 * scale))
    out['violations'] = viol
    out['holds'] = not any(viol.values())
    return out


def antiparallel(a):
    res = {}
    for base in ('T4', 'T5'):
        for e in 'LR':
            for x, y in (('a', 'b'), ('c', 'd')):
                p, q = a['D2'][f'{base}{x}_{e}'], a['D2'][f'{base}{y}_{e}']
                sep = abs((p['weighted_circular_mean_pref_deg'] - q['weighted_circular_mean_pref_deg'] + 180) % 360 - 180)
                d1p, d1q = a['D1'][f'{base}{x}_{e}']['preferred_deg'], a['D1'][f'{base}{y}_{e}']['preferred_deg']
                res[f'{base}{x}/{y}_{e}'] = dict(d2_pref=[p['weighted_circular_mean_pref_deg'],
                                                          q['weighted_circular_mean_pref_deg']],
                                                 d2_separation_deg=sep, d1_pref=[d1p, d1q],
                                                 antiparallel_d2_within_45deg=bool(sep >= 135),
                                                 antiparallel_d1=bool((d1p - d1q) % 360 == 180))
    return res


def summary(a, tag):
    if a is None:
        return None
    t45 = {k: dict(dsi=a['D1'][k]['dsi'], magnitude_mV=a['D1'][k]['magnitude'],
                   preferred_deg=a['D1'][k]['preferred_deg'],
                   d2_median=a['D2'][k]['median_cell_abs_dsi_f1'],
                   d2_frac_ge_0_2=a['D2'][k]['frac_cells_dsi_f1_ge_0_2'],
                   d2_pref=a['D2'][k]['weighted_circular_mean_pref_deg']) for k in T45}
    graded_drift = {k: v for k, v in a['instrument']['gray_drift_mV_second_minus_first_half'].items()
                    if not k.startswith(('DNa', 'PFL'))}
    m = meta[tag]
    return dict(
        gate=a['gate'], t4t5=t45, antiparallel=antiparallel(a),
        hs=a['D5'].get('sign_test'), dna02=a['D6'],
        brain_mean_hz={k: v['brain_mean_hz'] for k, v in a['scalars'].items()},
        graded_at_bound=dict(floor=max(v['graded_at_floor_frac'] for v in a['scalars'].values()),
                             ceiling=max(v['graded_at_ceiling_frac'] for v in a['scalars'].values())),
        instrument={k: v for k, v in a['instrument'].items() if k != 'gray_drift_mV_second_minus_first_half'},
        gray_drift_graded_max_abs_mV=max(abs(v) for v in graded_drift.values()),
        flicker_equivalent_delay_ms={k: round(v['equivalent_delay_ms'], 1) for k, v in a['D4'].items()
                                     if v['amplitude_mV'] >= 0.01},
        flicker_amplitude_mV={k: v['amplitude_mV'] for k, v in a['D4'].items()},
        wall_s=m.get('wall_s'), simulated_s=m.get('simulated_s'),
        simulated_s_per_wall_s=m.get('simulated_s_per_wall_s'))


S = {t: summary(a, t) for t, a in (('v4_protocol9', v4), ('v5_primary', v5), ('v5_upper_S2', s2))}

F = {}
F['F1_s0_not_bit_identical'] = dict(
    triggered=bool(s0['F1_triggered']),
    synthetic_cpu_and_gpu='bit-identical (tests/test_kinetics_v5.py)',
    real_graph=s0['v4_vs_v5_S0'], v4_vs_itself=s0['v4_vs_v4_repeat'], spikes=s0['spikes'],
    attribution='v4 is itself not bit-reproducible run to run on the real graph on this GPU; '
                'the v5-S0 difference equals v4\'s own run-to-run difference')
F['F2_unit_psp'] = dict(triggered=unit['F2_triggered'], checks=unit['F2_check'])
F['F3_gate_passes'] = dict(triggered=bool(v5 and v5['gate']['passed']))
F['F4_departs_from_v4'] = dict(primary=p3(v4, v5) if v5 else None, s2=p3(v4, s2, 2.0) if s2 else None)
F['F4_departs_from_v4']['triggered'] = bool(v5 and not F['F4_departs_from_v4']['primary']['holds'])
F['F4_departs_from_v4']['brain_mean_hz_gray'] = {t: S[t]['brain_mean_hz']['gray'] for t in S if S[t]}
inst = {}
for t, a in (('v4_protocol9', v4), ('v5_primary', v5), ('v5_upper_S2', s2)):
    if a:
        bad = []
        for k in ('R1-R6_L', 'R1-R6_R'):
            i = a['instrument'][k]
            if i['D1_dsi'] is not None and i['D1_dsi'] >= 0.1:
                bad.append(f'{k} D1 DSI {i["D1_dsi"]:.3f} at {i["D1_magnitude"]*1000:.2f} uV')
            if i['D2_median_cell_abs_dsi_f1'] >= 0.05:
                bad.append(f'{k} D2 median {i["D2_median_cell_abs_dsi_f1"]:.3f}')
        inst[t] = bad
F['F5_instrument_null_controls'] = dict(failures=inst, triggered=any(inst.values()))
F['F6_saturation_or_collapse'] = dict(
    triggered=any(S[t]['graded_at_bound']['floor'] > 0.5 or S[t]['graded_at_bound']['ceiling'] > 0.5
                  for t in S if S[t]),
    graded_at_bound={t: S[t]['graded_at_bound'] for t in S if S[t]})
F['F7_verdict_depends_on_arm'] = dict(
    triggered=bool(s2 and v5 and s2['gate']['passed'] != v5['gate']['passed']),
    primary=v5 and v5['gate']['passed'], s2=s2 and s2['gate']['passed'])

P = dict(
    P1=dict(holds=not unit['F2_triggered'], note='decay as declared, PSP area within 0.5 % of v4; '
            'S0 bit-identical on synthetic graphs, within v4\'s own run-to-run float noise on the real graph'),
    P2=dict(holds=bool(v5 and not v5['gate']['t4t5_clause'])),
    P3=dict(holds=bool(v5 and F['F4_departs_from_v4']['primary']['holds'])),
    P4=dict(holds=bool(v5 and not any(x['antiparallel_d2_within_45deg'] or x['antiparallel_d1']
                                      for x in S['v5_primary']['antiparallel'].values()))),
    P5=dict(holds=bool(v5 and all(abs(x['front_to_back_dc_mV']) < 1 and not (x['depolarises_ftb'] and x['hyperpolarises_btf'])
                                  for x in v5['D5']['sign_test'].values()))),
    P6=dict(holds=bool(v5 and not v5['D6']['yaw_clause']['passed'])),
    P7=dict(holds=bool(S['v5_primary'] and S['v4_protocol9']
                       and S['v5_primary']['simulated_s_per_wall_s'] >= S['v4_protocol9']['simulated_s_per_wall_s']),
            speed={t: S[t]['simulated_s_per_wall_s'] for t in S if S[t]}, pre_lock_probe=speed),
)

receipt = dict(
    spec='docs/LIF_DYNAMICS_SPEC.md#9',
    declaration_lock=LOCK,
    declaration_sha256=hashlib.sha256(open(LOCK, 'rb').read()).hexdigest(),
    dynamics='v5', kinetics_tables=unit['tables'],
    pins={t: meta[t]['dynamics_pin'] for t in meta},
    kinetics_reports={t: meta[t]['kinetics_report'] for t in meta},
    graded_report=meta['v5_primary']['graded_report'] if 'v5_primary' in meta else None,
    io_map=meta['v4_protocol9']['io_map'], encoder=meta['v4_protocol9']['encoder'],
    protocol=meta['v4_protocol9']['protocol'], yaw_axes=meta['v4_protocol9']['yaw_axes'],
    gate_verdict=dict(primary=v5 and v5['gate'], s2=s2 and s2['gate'], v4_reference=v4['gate'],
                      expensive_protocol_d_run=False if not (v5 and v5['gate']['passed']) else None),
    predictions=P, falsifiers=F, runs=S, unit_level=unit,
    raw_files=sorted(f for f in os.listdir(R)),
    posthoc_not_preregistered=dict(
        rate_locus=json.load(open(R + 'posthoc_rate_locus.json'))
        if os.path.exists(R + 'posthoc_rate_locus.json') else None,
        note='POST-HOC, not preregistered, cannot change any verdict: where the v5 primary\'s higher '
             'no-stimulus spike rate (F4) lives, 1 s of gray after 1 s settle, GTX 1660 Ti'),
    hardware_note='v4_protocol9, v5_primary, the a3 S0 check and the pre-lock speed probe ran on the '
                  'Quadro P620 (CUDA_VISIBLE_DEVICES=1 under the default FASTEST_FIRST ordering selects it); '
                  'v5_upper_S2 and the post-hoc diagnostic ran on the GTX 1660 Ti (CUDA_DEVICE_ORDER=PCI_BUS_ID). '
                  'Same arithmetic; only float-rounding and speed differ.',
    post_lock_corrections=[
        'analyse_tuning.py D4: f1_im is stored as +sum V sin(wt), the negative of the declared F1 '
        'imaginary part; the first version mirrored every lag. Fixed before any v5 analysis was read; '
        'definition unchanged; D1/D2/D5 unaffected.'],
)
json.dump(receipt, open('docs/receipts/lif_dynamics_v5.json', 'w'), indent=1, default=float)
print(json.dumps(dict(P={k: v['holds'] for k, v in P.items()},
                      F={k: v.get('triggered') for k, v in F.items()},
                      gate=receipt['gate_verdict']), indent=1, default=str)[:3000])
print('P3 violations primary', F['F4_departs_from_v4']['primary']['violations'] if v5 else None)
if s2:
    print('P3 violations s2', F['F4_departs_from_v4']['s2']['violations'])
