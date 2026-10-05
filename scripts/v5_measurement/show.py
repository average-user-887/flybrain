"""Print the D4/D5/D6/instrument summary of one analysis file."""
import json
import sys

a = json.load(open(f'docs/receipts/v5_raw/{sys.argv[1]}.analysis.json'))
print('scalars gray', a['scalars']['gray'])
print('instrument', {k: v for k, v in a['instrument'].items() if k != 'gray_drift_mV_second_minus_first_half'})
drift = a['instrument']['gray_drift_mV_second_minus_first_half']
print('gray drift max |mV|', max(abs(x) for x in drift.values()), max(drift, key=lambda k: abs(drift[k])))
print('D6', json.dumps({k: v for k, v in a['D6'].items()}, indent=0)[:2500])
for c in ('yaw_ccw', 'yaw_cw'):
    for k, v in a['D5'].get(c, {}).items():
        print(c, k, 'mean dc', round(v['mean_dc_mV'], 4), 'f1', [round(x, 3) for x in v['per_cell_f1_mV']],
              'gray', [round(x, 1) for x in v['gray_v_mV']])
print('HS sign', a['D5'].get('sign_test'))
for k in sorted(a['D4']):
    if k.endswith('_L'):
        d = a['D4'][k]
        print(f"D4 {k:9s} amp={d['amplitude_mV']:.3f} lag={d['phase_lag_deg']:.1f} {d['sign_vs_light']} "
              f"eqdelay={d['equivalent_delay_ms']:.1f}ms dc={d['dc_response_mV']:.4f}")
