import json
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1
                   else '<redacted-path>/tmp/graded/measure_bc.json'))
print('theta  DNa02_L gray/grating   DNa02_R gray/grating   L-R(grating)')
for t, dd in d['per_direction'].items():
    b, r = dd['baseline']['rate_hz'], dd['response']['rate_hz']
    print(f"{t:>5s}  {b['DNa02_L']:6.1f} / {r['DNa02_L']:6.1f}      "
          f"{b['DNa02_R']:6.1f} / {r['DNa02_R']:6.1f}      "
          f"{r['DNa02_L'] - r['DNa02_R']:+6.1f}")
print('\nbrain max single (2 ms bin artefact):',
      {t: dd['response']['brain_max_single_hz'] for t, dd in d['per_direction'].items()})
th = list(d['per_direction'])[0]
w = d['per_direction'][th]['response']
print('\nabsolute mean V, grating, theta', th)
for k in ('R1-R6_L', 'R1-R6_R', 'L1_L', 'L2_L', 'L3_L', 'L4_L', 'L5_L', 'Mi1_L', 'Mi4_L',
          'Mi9_L', 'Tm1_L', 'Tm9_L', 'CT1_L', 'T4a_L', 'T4b_L', 'T5a_L', 'HS_L', 'H2_L',
          'VS_L', 'DNa02_L', 'DNa02_R', 'PFL3_L'):
    if k in w['mean_v_mV']:
        print(f"  {k:9s} {w['mean_v_mV'][k]:8.3f} mV  (min {w['min_v_mV'][k]:7.2f} "
              f"max {w['max_v_mV'][k]:7.2f})")
