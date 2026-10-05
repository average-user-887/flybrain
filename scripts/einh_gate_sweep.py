"""Apply the LOCKED Q1/R1/R2 gate to every arm of the declared E_inh sweep.

Ported from the run directory outputs/wp5/einh-sweep-20260926/gate_sweep.py; only
the input/output location became an argument.  The criteria are copied verbatim
from outputs/wp5/v3-verify-20260926/gate.py,
which implements docs/LIF_DYNAMICS_SPEC.md §6.7.  Nothing about them is changed
here; only the set of configurations they are applied to is new, and that set
was fixed in docs/EINH_SENSITIVITY.md §2 before this ran.

Q1 quiet baseline : both gray windows, network rate < 1.0e5 spikes/s AND
                    DNa02_L and DNa02_R each < 20 Hz.
R1 responsiveness : during rotation, network rate > 0, the sign of the DNa02
                    L-R rate difference follows the stimulus in BOTH directions,
                    and |L-R| >= 1 Hz in at least one direction.
R2 bounded        : min V >= E_inh in every window.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / 'outputs/wp5/einh-sweep-20260926'

_parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_parser.add_argument('sweep_dir', nargs='?', default=str(DEFAULT_DIR),
                     help='directory holding probe_c_sweep.json (written by '
                          'scripts/lif_dynamics_diagnosis.py --e-inh-sweep); gate_sweep.json '
                          'is written next to it. Default: outputs/wp5/einh-sweep-20260926')
HERE = Path(_parser.parse_args().sweep_dir).resolve()

new = json.load(open(HERE / 'probe_c_sweep.json'))


def arm_key(row):
    mode = row.get('transmitter_policy', {}).get('unclear_mode', 'excitatory')
    e = row.get('e_inh_mV') or -70.0
    arm = 'A-primary' if mode == 'excitatory' else f'A-S1(unclear={mode})'
    return f'{arm} E_inh={e:g}'


def gate(rows):
    by_dir = {r['direction']: r for r in rows}
    out = {}
    q1 = True
    for d, r in by_dir.items():
        for w in ('initial_gray', 'gray_after_stimulus'):
            g = r[w]
            q1 &= (g['network_rate_hz'] < 1.0e5 and g['rate_l_hz'] < 20 and g['rate_r_hz'] < 20)
    diffs = {}
    r1_sign = True
    r1_mag = False
    for d, r in by_dir.items():
        s = r['stimulus']
        diff = s['rate_l_hz'] - s['rate_r_hz']
        diffs[d] = diff
        r1_sign &= (s['network_rate_hz'] > 0 and (diff * d) > 0)
        r1_mag |= abs(diff) >= 1.0

    def vmin(r):
        rng = r.get('network_membrane_range_mV')
        if rng is not None:
            return min(rng)
        return min(r[w][k] for w in ('initial_gray', 'stimulus', 'gray_after_stimulus')
                   for k in ('v_l_min', 'v_r_min'))

    r2 = all(vmin(r) >= (r.get('e_inh_mV') or -70.0) - 1e-6 for r in by_dir.values())
    out['membrane_min_mV'] = {str(d): vmin(r) for d, r in by_dir.items()}
    out.update(Q1=bool(q1), R1=bool(r1_sign and r1_mag), R2=bool(r2),
               R1_sign_both_directions=bool(r1_sign), R1_magnitude=bool(r1_mag),
               dna02_LminusR_per_direction={str(k): v for k, v in diffs.items()},
               dynamics_variant={str(d): r.get('dynamics_variant') for d, r in by_dir.items()},
               dynamics_variant_pin={str(d): r.get('dynamics_variant_pin')
                                     for d, r in by_dir.items()},
               gray=[{**{'direction': d, 'window': w}, **{k: r[w][k] for k in
                        ('network_rate_hz', 'rate_l_hz', 'rate_r_hz', 'v_l_min', 'v_r_min')}}
                     for d, r in by_dir.items() for w in ('initial_gray', 'gray_after_stimulus')],
               stimulus=[{**{'direction': d}, **{k: r['stimulus'][k] for k in
                            ('network_rate_hz', 'rate_l_hz', 'rate_r_hz')},
                          'membrane_range_mV': r.get('network_membrane_range_mV'),
                          'sim_s_per_wall_s': r['sim_s_per_wall_s']}
                         for d, r in by_dir.items()])
    return out


groups = {}
for r in new['probe_c']:
    groups.setdefault(arm_key(r), []).append(r)
arms = {k: gate(v) for k, v in groups.items()}

print('%-34s %-6s %-6s %-6s  %s' % ('configuration', 'Q1', 'R1', 'R2',
                                    'DNa02 L-R  (dir +1 / dir -1)'))
for key, g in arms.items():
    d = g['dna02_LminusR_per_direction']
    print('%-34s %-6s %-6s %-6s  %+7.1f / %+7.1f Hz%s'
          % (key, g['Q1'], g['R1'], g['R2'], d.get('1', float('nan')),
             d.get('-1', float('nan')),
             '' if g['R1'] else '   <-- R1 FAIL'))

print('\n--- gray windows (Q1) ---')
for key, g in arms.items():
    for w in g['gray']:
        print('%-34s dir%+d %-20s net %-10.4g L %5.1f R %5.1f  vmin L %.2f R %.2f'
              % (key, w['direction'], w['window'], w['network_rate_hz'], w['rate_l_hz'],
                 w['rate_r_hz'], w['v_l_min'], w['v_r_min']))

print('\n--- stimulus windows ---')
for key, g in arms.items():
    for s in g['stimulus']:
        rng = s['membrane_range_mV']
        print('%-34s dir%+d net %-10.4g L %5.1f R %5.1f  V [%.2f, %.2f]  %.4f sim s / wall s'
              % (key, s['direction'], s['network_rate_hz'], s['rate_l_hz'], s['rate_r_hz'],
                 rng[0], rng[1], s['sim_s_per_wall_s']))

sweep_fp = new.get('fixed_point_einh_sweep', {})
print('\n--- declared static fixed point across the sweep (arithmetic, no simulation) ---')
print('%-18s %-8s %-10s %-10s %-10s %-10s' % ('policy', 'E_inh', 'g_unit_inh', 'ratio', 'r', 'V*'))
for name, rows in sweep_fp.items():
    for key, rep in rows.items():
        print('%-18s %-8s %-10.6f %-10.3f %-10.4f %-10.3f'
              % (name, key, rep['g_unit_inh'], rep['g_unit_ratio_inh_over_exc'],
                 rep['conductance_ratio_g_inh_over_g_exc'],
                 rep['high_conductance_fixed_point_mV']))

out = dict(arms=arms, fixed_point=new['fixed_point'], fixed_point_einh_sweep=sweep_fp,
           graph=new['graph'], probe_c_policies=new.get('probe_c_policies'),
           prereg_sha256=new.get('prereg_sha256'), io_map_sha256=new.get('io_map_sha256'),
           dynamics_pins=new.get('dynamics_pins'), generated_at=new['generated_at'])
(HERE / 'gate_sweep.json').write_text(json.dumps(out, indent=2) + '\n')
print(f'\nwrote {HERE / "gate_sweep.json"}')
