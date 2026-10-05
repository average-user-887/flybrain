"""Emit the markdown tables for WP5 §13 from the measurement JSONs."""
import json
import sys

T = '<redacted-path>/tmp/graded/'
v4 = json.load(open(T + 'measure_bc.json'))
try:
    v3 = json.load(open(T + 'measure_bc_v3.json'))
except FileNotFoundError:
    v3 = None
try:
    s1 = json.load(open(T + 'measure_bc_s1.json'))
except FileNotFoundError:
    s1 = None
try:
    tr4 = json.load(open(T + 'trace_v4.json'))
    tr3 = json.load(open(T + 'trace_v3.json'))
except FileNotFoundError:
    tr4 = tr3 = None

order = ['R1-R6', 'L1', 'L2', 'L3', 'L4', 'L5', 'Mi1', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm3',
         'Tm4', 'Tm9', 'CT1', 'T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d',
         'HS', 'H2', 'VS', 'DNa02']

print('### DSI table (v4 primary)\n')
print('| population | eye | n | graded | response magnitude | preferred dir | DSI |')
print('|---|---|---|---|---|---|---|')
for name in order:
    for eye in ('L', 'R'):
        k = f'{name}_{eye}'
        if k not in v4['direction_selectivity']:
            continue
        e = v4['direction_selectivity'][k]
        n = v4['populations'][k]['n']
        gr = 'graded' if v4['populations'][k]['graded'] else 'spiking'
        ds = '—' if e['dsi'] is None else f"{e['dsi']:.3f}"
        print(f"| {name} | {eye} | {n} | {gr} | {e['magnitude']:.4f} {e['unit']} | "
              f"{e['preferred_deg']:.0f}° | {ds} |")

if tr4 and tr3:
    th = list(tr4['per_direction'])[0]
    print(f'\n### per-cell trace, direction {th}deg: v3 vs v4\n')
    print('| population | V gray v3 | V gray v4 | DC v3 | DC v4 | |DC| v4 | mod gray v4 | '
          'mod grating v4 | spikes v3 | spikes v4 |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for name in order:
        for eye in ('L',):
            k = f'{name}_{eye}'
            if k not in tr4['per_direction'][th]:
                continue
            a = tr3['per_direction'][th][k]
            b = tr4['per_direction'][th][k]
            print(f"| {name}_{eye} | {a['v_gray_mean']:.3f} | {b['v_gray_mean']:.3f} | "
                  f"{a['dc_mean_mV']:+.4f} | {b['dc_mean_mV']:+.4f} | {b['dc_abs_mean_mV']:.4f} | "
                  f"{b['modulation_gray_mV']:.4f} | {b['modulation_grating_mV']:.4f} | "
                  f"{a['spikes_grating']} | {b['spikes_grating']} |")
    for tag, tr in (('v3', tr3), ('v4', tr4)):
        br = tr['per_direction'][th]['__brain__']
        print(f"\n{tag} brain: mean {br['mean_hz_grating']:.3f} Hz, max single "
              f"{br['max_single_hz_grating']:.1f} Hz, active {br['active_neurons_grating']}, "
              f"graded at floor {br['graded_at_floor_frac']:.4f} ceiling "
              f"{br['graded_at_ceiling_frac']:.4f}, V range {br['v_min_mV']:.2f}..{br['v_max_mV']:.2f}")

if s1:
    print('\n### arm S1 (tier-1 graded only) vs primary: T4/T5 DSI\n')
    print('| population | primary magnitude | primary DSI | S1 magnitude | S1 DSI |')
    print('|---|---|---|---|---|')
    for name in order:
        for eye in ('L', 'R'):
            k = f'{name}_{eye}'
            if not name.startswith(('T4', 'T5', 'HS', 'DNa02')):
                continue
            if k not in v4['direction_selectivity'] or k not in s1['direction_selectivity']:
                continue
            a, b = v4['direction_selectivity'][k], s1['direction_selectivity'][k]
            fa = '—' if a['dsi'] is None else f"{a['dsi']:.3f}"
            fb = '—' if b['dsi'] is None else f"{b['dsi']:.3f}"
            print(f"| {name}_{eye} | {a['magnitude']:.4f} | {fa} | {b['magnitude']:.4f} | {fb} |")
if v3:
    print('\n### v3 like-for-like, same encoder and protocol\n')
    for name in ('R1-R6', 'L1', 'L2', 'L3', 'T4a', 'T5a', 'HS', 'DNa02'):
        for eye in ('L',):
            k = f'{name}_{eye}'
            if k in v3['direction_selectivity']:
                e = v3['direction_selectivity'][k]
                print(f"  {k}: magnitude {e['magnitude']:.4f} {e['unit']}, DSI "
                      f"{e['dsi'] if e['dsi'] is None else round(e['dsi'], 3)}")
    d0 = list(v3['per_direction'])[0]
    for w in ('baseline', 'response'):
        ww = v3['per_direction'][d0][w]
        print(f"  v3 {w}: brain_mean {ww['brain_mean_hz']:.4f} Hz; "
              f"L1_L max {ww['max_v_mV']['L1_L']:.4f} min {ww['min_v_mV']['L1_L']:.4f}; "
              f"R1-R6_L rate {ww['rate_hz']['R1-R6_L']:.2f} Hz; "
              f"DNa02_L rate {ww['rate_hz']['DNa02_L']:.2f} Hz")
