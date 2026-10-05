import json
import sys

T = '<redacted-path>/tmp/graded/'
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else T + 'measure_bc.json'))
dsi = d['direction_selectivity']
print('dynamics', d['dynamics'], '| backend', d['backend'],
      '| graded', (d['graded_report'] or {}).get('graded_neurons', 0),
      '| wall', round(d['wall_s']), 's for', d['simulated_s'], 's simulated',
      '| speed %.4f sim-s per wall-s' % (d['simulated_s'] / d['wall_s']))

order = ['R1-R6', 'L1', 'L2', 'L3', 'L4', 'L5', 'Mi1', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm3',
         'Tm4', 'Tm9', 'CT1', 'T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d',
         'HS', 'H2', 'VS', 'DNa02', 'DNa01', 'PFL3']
print('\n%-9s %-4s %5s %9s %9s %6s %7s  tuning (deg: response)' %
      ('pop', 'eye', 'n', 'magnitude', 'pref_resp', 'pref', 'DSI'))
for name in order:
    for eye in ('L', 'R'):
        k = f'{name}_{eye}'
        if k not in dsi:
            continue
        e = dsi[k]
        n = d['populations'][k]['n']
        ds = 'n/a' if e['dsi'] is None else f"{e['dsi']:.3f}"
        tun = ' '.join(f"{int(float(t)):3d}:{v:+.4f}" for t, v in e['tuning'].items())
        print(f"{name:9s} {eye:4s} {n:5d} {e['magnitude']:9.4f} {e['r_pref']:9.4f} "
              f"{e['preferred_deg']:6.0f} {ds:>7s}  {tun}")

print('\n-- network state, per direction (gray | grating) --')
for theta, dd in d['per_direction'].items():
    b, r = dd['baseline'], dd['response']
    print(f"theta {theta:>4s}: brain_mean {b['brain_mean_hz']:.3f} | {r['brain_mean_hz']:.3f} Hz"
          f"  graded_at_floor {b['graded_at_floor_frac']:.5f} | {r['graded_at_floor_frac']:.5f}"
          f"  at_ceiling {b['graded_at_ceiling_frac']:.5f} | {r['graded_at_ceiling_frac']:.5f}")

print('\n-- HS physiology (measurement b) --')
for theta, dd in d['per_direction'].items():
    b, r = dd['baseline'], dd['response']
    hs = [round(x - y, 3) for x, y in zip(r['hs_v_mV'], b['hs_v_mV'])]
    print(f"theta {theta:>4s}: HS dV mV (L1-4 then R1-4) {hs}   spikes L {r['spikes']['HS_L']} "
          f"R {r['spikes']['HS_R']}")
print('HS absolute V in gray (theta 0):', [round(x, 2) for x in
      d['per_direction'][list(d['per_direction'])[0]]['baseline']['hs_v_mV']])

print('\n-- lamina: is the max still pinned at -52.00 (the v3 signature)? --')
first = d['per_direction'][list(d['per_direction'])[0]]
for k in ('L1_L', 'L1_R', 'L2_L', 'L3_L'):
    print(f"  {k}: gray max {first['baseline']['max_v_mV'][k]:.4f} min "
          f"{first['baseline']['min_v_mV'][k]:.4f} | grating max "
          f"{first['response']['max_v_mV'][k]:.4f} min {first['response']['min_v_mV'][k]:.4f}")

print('\n-- DNa02 --')
for k in ('DNa02_L', 'DNa02_R'):
    rates = {t: dd['response']['rate_hz'][k] for t, dd in d['per_direction'].items()}
    vs = {t: round(dd['response']['mean_v_mV'][k], 3) for t, dd in d['per_direction'].items()}
    print(' ', k, 'rates', rates)
    print(' ', k, 'meanV', vs)
