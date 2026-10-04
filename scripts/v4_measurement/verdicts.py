"""Evaluate the predeclared falsification criteria F1-F7 and the gate on (d)."""
import json

T = '<redacted-path>/tmp/graded/'
a = json.load(open(T + 'measure_a.json'))
v4 = json.load(open(T + 'measure_bc.json'))


def load(name):
    try:
        return json.load(open(T + name))
    except FileNotFoundError:
        return None


v3 = load('measure_bc_v3.json')
s1 = load('measure_bc_s1.json')
tr4 = load('trace_v4.json')
tr3 = load('trace_v3.json')

out = {}

# F1: no subthreshold transmission
lam_pinned = None
if tr4:
    th = list(tr4['per_direction'])[0]
    lam_pinned = all(abs(tr4['per_direction'][th][k]['dc_mean_mV']) < 1e-6
                     for k in ('L1_L', 'L2_L', 'L3_L'))
out['F1_no_subthreshold_transmission'] = dict(
    unit_level=a['a1_verdict'],
    lamina_dc_all_zero=lam_pinned,
    triggered=bool(a['a1_verdict'].startswith('FAIL') or lam_pinned))

# F2 / F3: saturation or collapse, evaluated in the no-stimulus window
floors, ceils, means = [], [], []
for d in v4['per_direction'].values():
    floors.append(d['baseline']['graded_at_floor_frac'])
    ceils.append(d['baseline']['graded_at_ceiling_frac'])
    means.append(d['baseline']['brain_mean_hz'])
out['F2_graded_saturates'] = dict(max_graded_at_ceiling_frac=max(ceils),
                                  max_brain_mean_hz_gray=max(means),
                                  triggered=bool(max(ceils) > 0.5))
out['F3_graded_collapses'] = dict(max_graded_at_floor_frac=max(floors),
                                  triggered=bool(max(floors) > 0.5))

# F4: S0 bit-identity
out['F4_s0_not_bit_identical'] = dict(check=a['a3_verdict'],
                                      triggered=not a['a3_verdict'].startswith('PASS'))

# F5: HS response window 0.1-20 mV, and the sign
hs = {}
for theta, d in v4['per_direction'].items():
    dv = [r - b for r, b in zip(d['response']['hs_v_mV'], d['baseline']['hs_v_mV'])]
    hs[theta] = dict(per_cell_dV_mV=[round(x, 4) for x in dv],
                     max_abs=max(abs(x) for x in dv),
                     spikes=d['response']['spikes']['HS_L'] + d['response']['spikes']['HS_R'])
best = max(hs.values(), key=lambda x: x['max_abs'])['max_abs']
out['F5_HS_outside_window'] = dict(
    best_abs_response_mV=best, per_direction=hs,
    hs_spikes_total=sum(v['spikes'] for v in hs.values()),
    triggered=bool(best < 0.1 or best > 20.0))

# F6: no direction selectivity in T4/T5
t45 = {k: v for k, v in v4['direction_selectivity'].items() if k[:2] in ('T4', 'T5')}
strong = {k: v for k, v in t45.items()
          if v['dsi'] is not None and v['dsi'] >= 0.2 and v['magnitude'] >= 0.5}
pairs = {}
for base in ('T4', 'T5'):
    for eye in ('L', 'R'):
        a_, b_ = t45.get(f'{base}a_{eye}'), t45.get(f'{base}b_{eye}')
        c_, d_ = t45.get(f'{base}c_{eye}'), t45.get(f'{base}d_{eye}')
        if a_ and b_:
            pairs[f'{base}a/b_{eye}'] = dict(
                pref_a=a_['preferred_deg'], pref_b=b_['preferred_deg'],
                separation_deg=min((a_['preferred_deg'] - b_['preferred_deg']) % 360,
                                   (b_['preferred_deg'] - a_['preferred_deg']) % 360),
                antiparallel=bool((a_['preferred_deg'] - b_['preferred_deg']) % 360 == 180))
        if c_ and d_:
            pairs[f'{base}c/d_{eye}'] = dict(
                pref_c=c_['preferred_deg'], pref_d=d_['preferred_deg'],
                antiparallel=bool((c_['preferred_deg'] - d_['preferred_deg']) % 360 == 180))
out['F6_no_direction_selectivity'] = dict(
    t4t5_dsi={k: (None if v['dsi'] is None else round(v['dsi'], 3)) for k, v in t45.items()},
    t4t5_magnitude_mV={k: round(v['magnitude'], 4) for k, v in t45.items()},
    subtypes_passing_dsi_0_2_and_0_5mV=sorted(strong),
    subtype_pairs=pairs,
    triggered=bool(not strong))

# F7: does the arm change the answer?
if s1:
    s1t = {k: v for k, v in s1['direction_selectivity'].items() if k[:2] in ('T4', 'T5')}
    s1_strong = {k for k, v in s1t.items()
                 if v['dsi'] is not None and v['dsi'] >= 0.2 and v['magnitude'] >= 0.5}
    out['F7_depends_on_graded_class_list'] = dict(
        primary_passing=sorted(strong), s1_passing=sorted(s1_strong),
        triggered=bool(bool(strong) != bool(s1_strong)))
else:
    out['F7_depends_on_graded_class_list'] = dict(triggered=None, note='arm S1 not run')

# the declared gate on (d)
dn_signal = {}
for k in ('DNa02_L', 'DNa02_R'):
    e = v4['direction_selectivity'].get(k)
    if e:
        dn_signal[k] = dict(magnitude_hz=e['magnitude'], tuning=e['tuning'])
diff = []
for theta, d in v4['per_direction'].items():
    diff.append(dict(theta=theta,
                     l=d['response']['rate_hz']['DNa02_L'],
                     r=d['response']['rate_hz']['DNa02_R'],
                     abs_diff=abs(d['response']['rate_hz']['DNa02_L']
                                  - d['response']['rate_hz']['DNa02_R'])))
out['gate_on_d'] = dict(
    t4t5_criterion_met=bool(strong),
    dna02_rates=diff,
    dna02_criterion_met=bool(any(x['abs_diff'] >= 1.0 for x in diff)),
    passed=bool(strong) and bool(any(x['abs_diff'] >= 1.0 for x in diff)),
    rule='spec §7.10: DSI >= 0.2 in a T4/T5 subtype with magnitude >= 0.5 mV, AND '
         '|rate(DNa02_L) - rate(DNa02_R)| >= 1 Hz with the sign following the stimulus')

print(json.dumps(out, indent=2))
with open(T + 'verdicts.json', 'w') as fh:
    json.dump(out, fh, indent=2)
