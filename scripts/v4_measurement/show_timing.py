from _paths import REPO, SCRATCH  # repo root; scratch = $NEUROFLY_V4_SCRATCH or outputs/v4_measurement
import json
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1
                   else str(SCRATCH / 'timing.json')))
print('backend', d['backend'], 'graded', d['graded_report']['graded_neurons'],
      'load_s', round(d['graph_load_s'], 1), 'wall_s', round(d['wall_s'], 1),
      'sim_s', d['simulated_s'])
k0 = list(d['per_direction'])[0]
b = d['per_direction'][k0]['baseline']
r = d['per_direction'][k0]['response']
for tag, w in (('gray', b), ('grating', r)):
    print(f"\n== {tag}: brain_mean {w['brain_mean_hz']:.3f} Hz, brain_max_single "
          f"{w['brain_max_single_hz']:.0f} Hz, graded at floor {w['graded_at_floor_frac']:.4f}, "
          f"at ceiling {w['graded_at_ceiling_frac']:.4f}")
    for k in ('R1-R6_L', 'L1_L', 'L2_L', 'L3_L', 'L5_L', 'Mi1_L', 'Mi4_L', 'Mi9_L',
              'Tm1_L', 'Tm9_L', 'CT1_L', 'T4a_L', 'T5a_L', 'HS_L', 'H2_L', 'VS_L',
              'DNa02_L', 'DNa02_R', 'PFL3_L'):
        if k in w['mean_v_mV']:
            print(f"  {k:9s} meanV {w['mean_v_mV'][k]:8.3f}  min {w['min_v_mV'][k]:8.3f}  "
                  f"max {w['max_v_mV'][k]:8.3f}  rate {w['rate_hz'][k]:8.2f} Hz")
print('\nHS per-cell V (gray / grating):')
print([round(x, 3) for x in b['hs_v_mV']])
print([round(x, 3) for x in r['hs_v_mV']])
